from diffusers import FluxPipeline
import torch
import torch.distributed as dist
import time

from para_attn.context_parallel import init_context_parallel_mesh
from para_attn.context_parallel.diffusers_adapters import parallelize_pipe
from para_attn.parallel_vae.diffusers_adapters import parallelize_vae

class FluxPredictor:

    def setup(self):
        dist.init_process_group(backend='nccl')
        torch.cuda.set_device(dist.get_rank())

        self.pipe = FluxPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-dev",
            torch_dtype=torch.bfloat16,
        ).to("cuda")

        mesh = init_context_parallel_mesh(
            self.pipe.device.type,
            max_ring_dim_size=2,
        )
        parallelize_pipe(
            self.pipe,
            mesh=mesh,
        )
        parallelize_vae(self.pipe.vae, mesh=mesh._flatten())
            
    def predict(self, 
        prompt: str,
        output_path: str,
        negative_prompt: str,
        guidance_scale: float,
        num_inference_steps: int,
    ):
        image = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            output_type="pil" if dist.get_rank() == 0 else "pt",
        ).images[0]

        if dist.get_rank() == 0:
            image.save(output_path)

        return image
    

    def get_test_predict_args(self):
        """
        helpful for unit testing
        """
        return {
            "prompt": "A cat in a hat",
            "output_path": "cat_in_hat.png",
            "negative_prompt": "",
            "guidance_scale": 7,
            "num_inference_steps": 25
        }

######################################
# some mock methods for unit testing #
######################################

class MockPipeline:

    def __call__(self, prompt, output_type="pil"):
        self._interrupt = False
        for i in range(10):
            if self._interrupt:
                continue
            time.sleep(1)

class MockPredictor:
    
    def setup(self):
        dist.init_process_group()
        torch.cuda.set_device(dist.get_rank())
        self.pipe = MockPipeline()

    def predict(self, prompt, output_path):
        self.pipe(prompt)
        with open(output_path, "w") as f:
            f.write("hello")
    
    def get_test_predict_args(self):
        return {
            "prompt": "A dog in a toupee",
            "output_path": "dog_in_toupee.png"
        }

class MockPredictorSetupError(MockPredictor):

    def setup(self):
        dist.init_process_group()
        torch.cuda.set_device(dist.get_rank())
        raise Exception("MockPredictorSetupError")


class MockPredictorPredictError(MockPredictor):

    def predict(self, prompt, output_path):
        for i in range(10):
            if i == 5:
                raise Exception("MockPredictorPredictError")
            time.sleep(1)
