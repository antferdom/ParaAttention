import torch
import torch.distributed as dist
from diffusers import FluxPipeline

dist.init_process_group()

pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16).to(
    f"cuda:{dist.get_rank()}"
)

from para_attn.context_parallel import init_context_parallel_mesh
from para_attn.context_parallel.diffusers_adapters import parallelize_pipe

parallelize_pipe(
    pipe,
    mesh=init_context_parallel_mesh(
        pipe.device.type,
        max_ring_dim_size=2,
    ),
)

torch._inductor.config.reorder_for_compute_comm_overlap = True
pipe.transformer = torch.compile(pipe.transformer, mode="max-autotune-no-cudagraphs")

image = pipe("A cat holding a sign that says hello world", num_inference_steps=28).images[0]

if dist.get_rank() == 0:
    image.save("flux.png")

dist.destroy_process_group()
