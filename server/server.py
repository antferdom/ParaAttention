import base64
import logging
import os
import pickle
import threading
import time
from collections import deque
from contextlib import asynccontextmanager
from queue import Queue
from typing import AsyncGenerator, Callable, Optional
import argparse

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import uvicorn
from diffusers import FluxPipeline

from flask import Flask, jsonify, request
from PIL import Image
from typing_extensions import TypeAlias  # Python 3.10+

from para_attn.context_parallel import init_context_parallel_mesh
from para_attn.context_parallel.diffusers_adapters import parallelize_pipe
from para_attn.parallel_vae.diffusers_adapters import parallelize_vae

dist.init_process_group()
# Set NCCL timeout and error handling
os.environ["NCCL_BLOCKING_WAIT"] = "1"
os.environ["NCCL_ASYNC_ERROR_HANDLING"] = "1"

app = Flask(__name__)
# Global variables
pipe = None
engine_config = None
input_config = None
local_rank = None
logger = None
initialized = False
args = None

# a global queue to store request prompts
request_queue = deque()
queue_lock = threading.Lock()
queue_event = threading.Event()
results_store = {}  # store request results

def setup_logger():
    global logger
    rank = dist.get_rank()
    logging.basicConfig(
        level=logging.INFO,
        format=f"[Rank {rank}] %(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logger = logging.getLogger(__name__)


@app.route("/initialize", methods=["GET"])
def check_initialize():
    global initialized
    if initialized:
        return jsonify({"status": "initialized"}), 200
    else:
        return jsonify({"status": "initializing"}), 202


def initialize():
    global pipe, local_rank, initialized, args
    mp.set_start_method("spawn", force=True)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default="black-forest-labs/FLUX.1-dev",
    )
    parser.add_argument(
        "--ulysses_degree",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--ring_degree",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--height",
        type=int,
        default=1024,
    )
    parser.add_argument(
        "--width",
        type=int,
        default=1024,
    )
    parser.add_argument(
        "--max_queue_size",
        type=int,
        default=4,
    )
    args = parser.parse_args()
    setup_logger()

    device = torch.device(f"cuda:{dist.get_rank()}")
    logger.info(f"Initializing model on GPU: {device}")

    pipe = FluxPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-dev",
        torch_dtype=torch.bfloat16,
    ).to(device)

    mesh = init_context_parallel_mesh(
        pipe.device.type,
        max_ring_dim_size=args.ulysses_degree,
    )
    parallelize_pipe(
        pipe,
        mesh=mesh,
    )
    parallelize_vae(pipe.vae, mesh=mesh._flatten())

    torch._inductor.config.reorder_for_compute_comm_overlap = True
    pipe.transformer = torch.compile(pipe.transformer, mode="max-autotune-no-cudagraphs")

    setup_logger()
    logger.info("Model initialization completed")
    initialized = True  # Set initialization completion flag


def generate_image_parallel(
    prompt,
    num_inference_steps: int,
    seed: int,
    cfg: float, 
    save_disk_path: Optional[str] = None,
):
    global pipe, local_rank
    logger.info(f"Starting image generation with prompt: {prompt}")
    torch.cuda.reset_peak_memory_stats()
    start_time = time.time()
    output = pipe(
        "A cat holding a sign that says hello world",
        num_inference_steps=num_inference_steps,
        output_type="pil",
    )
    image = output.images[0]
    end_time = time.time()
    elapsed_time = end_time - start_time
    logger.info(f"Image generation completed in {elapsed_time:.2f} seconds")

    if save_disk_path is not None:
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        filename = f"generated_image_{timestamp}.png"
        file_path = os.path.join(save_disk_path, filename)
        is_dp_last_group = dist.get_rank() == dist.get_world_size() - 1
        if is_dp_last_group:
            os.makedirs(save_disk_path, exist_ok=True)
            image.save(file_path)
            logger.info(f"Image saved to: {file_path}")
    # single gpu didn't need to distribute
    elif dist.get_world_size() > 1:
        if dist.get_rank() == 0:
            # serialize output object
            output_bytes = pickle.dumps(image)

            # send output to rank 0
            dist.send(
                torch.tensor(len(output_bytes), device=f"cuda:{local_rank}"), dst=0
            )
            dist.send(
                torch.ByteTensor(list(output_bytes)).to(f"cuda:{local_rank}"), dst=0
            )

            logger.info("Output sent to rank 0")

        if dist.get_rank() == 0:
            # recv from rank world_size - 1
            size = torch.tensor(0, device=f"cuda:{local_rank}")
            dist.recv(size, src=dist.get_world_size() - 1)
            output_bytes = torch.ByteTensor(size.item()).to(f"cuda:{local_rank}")
            dist.recv(output_bytes, src=dist.get_world_size() - 1)

            # deserialize output object
            output = pickle.loads(output_bytes.cpu().numpy().tobytes())

    return output, elapsed_time


@app.route("/generate", methods=["POST"])
def queue_image_request():
    logger.info("Received POST request for image generation")
    data = request.json
    request_id = str(time.time())
    
    with queue_lock:
        request_params = {
            "id": request_id,
            "prompt": data.get("prompt", "A cat holding a sign that says hello world"),
            "num_inference_steps": data.get("num_inference_steps", 28),
            "seed": data.get("seed", 1),
            "cfg": data.get("cfg", 8.0),
            "save_disk_path": data.get("save_disk_path")
        }
        
        request_queue.append(request_params)
        queue_event.set()
    
    return jsonify({
        "message": "Request accepted",
        "request_id": request_id,
        "status_url": f"/status/{request_id}"
    }), 202

@app.route("/status/<request_id>", methods=["GET"])
def check_status(request_id):
    if request_id in results_store:
        result = results_store.pop(request_id) 
        return jsonify(result), 200
    
    position = None
    with queue_lock:
        for i, req in enumerate(request_queue):
            if req["id"] == request_id:
                position = i
                break
    
    if position is not None:
        return jsonify({
            "status": "pending",
            "queue_position": position
        }), 202
    
    return jsonify({"status": "not_found"}), 404

def process_queue():
    while True:
        queue_event.wait()
        
        with queue_lock:
            if not request_queue:
                queue_event.clear()
                continue
            
            params = request_queue.popleft()
            if not request_queue:
                queue_event.clear()
        
        try:
            # Extract parameters
            request_id = params["id"]
            prompt = params["prompt"]
            num_inference_steps = params["num_inference_steps"]
            seed = params["seed"]
            cfg = params["cfg"]
            save_disk_path = params["save_disk_path"]
            
            # Broadcast parameters to all processes
            broadcast_params = [prompt, num_inference_steps, seed, cfg, save_disk_path]
            dist.broadcast_object_list(broadcast_params, src=0)
            
            # Generate image and get results
            output, elapsed_time = generate_image_parallel(*broadcast_params)
            
            # Process output results
            if save_disk_path:
                # output is disk path
                result = {
                    "message": "Image generated successfully",
                    "elapsed_time": f"{elapsed_time:.2f} sec",
                    "output": output,  # This is the file path
                    "save_to_disk": True
                }
            else:
                # Process base64 output
                if output and hasattr(output, "images") and output.images:
                    output_base64 = base64.b64encode(output.images[0].tobytes()).decode("utf-8")
                else:
                    output_base64 = ""
                
                result = {
                    "message": "Image generated successfully",
                    "elapsed_time": f"{elapsed_time:.2f} sec",
                    "output": output_base64,
                    "save_to_disk": False
                }
            
            # Store results
            results_store[request_id] = result
            
        except Exception as e:
            logger.error(f"Error processing request {params['id']}: {str(e)}")
            results_store[request_id] = {
                "error": str(e),
                "status": "failed"
            }


def run_host():
    if dist.get_rank() == 0:
        logger.info("Starting Flask host on rank 0")
        # process 0 will process the queue in a separate thread
        queue_thread = threading.Thread(target=process_queue, daemon=True)
        queue_thread.start()
        app.run(host="0.0.0.0", port=6000)
    else:
        while True:
            # Non-master processes wait for broadcasted parameters
            params = [None] * 5
            logger.info(f"Rank {dist.get_rank()} waiting for tasks")
            dist.broadcast_object_list(params, src=0)
            if params[0] is None:
                logger.info("Received exit signal, shutting down")
                break
            logger.info(f"Received task with parameters: {params}")
            generate_image_parallel(*params)


if __name__ == "__main__":
    """
        Request example:
        curl -X POST http://127.0.0.1:6000/generate \
            -H "Content-Type: application/json" \
            -d '{
                    "prompt": "A tree",
                    "num_inference_steps": 28,
                    "seed": 42,
                    "save_disk_path": "output"
                }'
    """
    initialize()

    logger.info(
        f"Process initialized. Rank: {dist.get_rank()}, Local Rank: {os.environ.get('LOCAL_RANK', 'Not Set')}"
    )
    logger.info(f"Available GPUs: {torch.cuda.device_count()}")
    run_host()