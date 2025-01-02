import logging
import os
from typing import Callable, List, Tuple
from PIL import Image


import torch
import torch._inductor.config as inductor_config
import torch.distributed as dist
import torch.nn as nn
import typer
from diffusers import FluxPipeline

from para_attn.context_parallel import init_context_parallel_mesh
from para_attn.context_parallel.diffusers_adapters import parallelize_pipe
from para_attn.parallel_vae.diffusers_adapters import parallelize_vae

dist.init_process_group(backend="nccl")
app = typer.Typer()


def benchmark_torch_function(iters, f, *args, **kwargs):
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    # Warmup runs
    for _ in range(2):
        f(*args, **kwargs)
    torch.cuda.synchronize()
    dist.barrier()  # Ensure all ranks finished warmup before starting timing
    
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    for _ in range(iters):
        output = f(*args, **kwargs)
    end_event.record()
    torch.cuda.synchronize()
    dist.barrier()
    
    # elapsed_time has a resolution of 0.5 microseconds:
    # but returns milliseconds, so we need to multiply it to increase resolution
    local_avg_time_us = start_event.elapsed_time(end_event) * 1000 / iters

    time = torch.tensor([local_avg_time_us], dtype=torch.float32, device=f"cuda:{rank}")
    dist.all_reduce(time, op=dist.ReduceOp.AVG)
    global_avg_us = time.item()
    return global_avg_us, output


def _compile_transformer_backbone(
    transformer: nn.Module,
    fullgraph: bool = True,
    debug: bool = False,
):
    if debug:
        os.environ["TORCH_COMPILE_DEBUG"] = "1"
        os.environ["TORCH_LOGS"] = "+inductor,dynamo"
        inductor_config.debug = True
        # Whether to disable a progress bar for autotuning
        inductor_config.disable_progress = False
        # Whether to enable printing the source code for each future
        inductor_config.verbose_progress = True
        inductor_config.trace.enabled = True
        inductor_config.trace.debug_log = True
        inductor_config.trace.info_log = True

    # torch._inductor.list_options()
    inductor_config.conv_1x1_as_mm = True  # treat 1x1 convolutions as matrix muls
    inductor_config.max_autotune_gemm_backends = "ATEN,TRITON"

    inductor_config.cuda.compile_opt_level = "-O3"  # default: "-O1"
    inductor_config.cuda.use_fast_math = True

    if getattr(transformer, "forward") is not None:
        optimized_transformer_forward = torch.compile(
            getattr(transformer, "forward"),
            fullgraph=fullgraph,
            backend="inductor",
            mode="max-autotune-no-cudagraphs",
        )
        setattr(transformer, "forward", optimized_transformer_forward)
    else:
        raise AttributeError(
            f"Transformer backbone type: {transformer.__class__.__name__} has no attribute 'forward'"
        )
    return transformer

@app.command()
def main(
    model_id: str = "black-forest-labs/FLUX.1-dev",
    height: int = 1024,
    width: int = 1024,
    diffusion_steps: int = 28,
    max_sequence_length: int = 512,
    use_torch_compile: bool = True,
    fullgraph: bool = True,
    debug: bool = True,
    iters: int = 10,
    seed: int = 42,   
):
    pipe = FluxPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
    ).to(f"cuda:{dist.get_rank()}")

    mesh = init_context_parallel_mesh(
        pipe.device.type,
        max_ring_dim_size=4,
    )
    parallelize_pipe(
        pipe,
        mesh=mesh,
    )
    parallelize_vae(pipe.vae, mesh=mesh._flatten())

    inductor_config.reorder_for_compute_comm_overlap = True

    transformer = getattr(pipe, "transformer", None)
    if transformer is not None and use_torch_compile:
       pipe.transformer = torch.compile(pipe.transformer, mode="max-autotune-no-cudagraphs")


    forward_time, output = benchmark_torch_function(
        iters,
        pipe,
        height=height,
        width=width,
        prompt="A tree in the forest",
        num_inference_steps=diffusion_steps,
        max_sequence_length=max_sequence_length,
        guidance_scale=0.0,
        generator=torch.Generator(device=f"cuda:{dist.get_rank()}").manual_seed(seed),
        output_type="pil" if dist.get_rank() == 0 else "pt",
    )

    if dist.get_rank() == 0:
        # import code; code.interact(local=locals())
        image = output.images[0]
        print(f"FLUX fwd time: {forward_time*1e-6:2.4f} s")
        print("Saving image to flux.png")
        image.save("flux.png")

    dist.barrier()
    dist.destroy_process_group()

    

if __name__ == "__main__":
    typer.run(main)