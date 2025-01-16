from typing import Optional

import importlib
import torch
import typer
import torch.multiprocessing as mp

from para_attn.server import AsyncGenerationServer

current_method = mp.get_start_method(allow_none=True)
if current_method != "spawn":
    mp.set_start_method('spawn', force=True)

app = typer.Typer()

def load_model(model_path: str):
    module_name, class_name = model_path.rsplit(":", 1)
    module = importlib.import_module(module_name)
    return getattr(module, class_name)

@app.command()
def serve(
    model: str = typer.Option(..., "--model", help="The model to serve"),
    host: Optional[str] = typer.Option("localhost", "--host", help="The hostname to serve the model on"),
    port: Optional[int] = typer.Option(5000, "--port", help="The port number to serve the model on"),
    num_devices: Optional[int] = typer.Option(..., "--num_devices", help="The number of devices to serve the model on"),
):
    """
    Serve the model on a specified host and port.

    Parameters:
    - model (str): module path of the model, in the format of module.submodule:ModelClassName
    - host (Optional[str]): The hostname to serve the model on. Defaults to "127.0.0.1".
    - port (Optional[int]): The port number to serve the model on. Defaults to 5000.
    - num_devices (Optional[int]): The number of devices to serve the model on. Defaults to all devices.
    """
    model_class = load_model(model)
    if num_devices is None:
        num_devices = torch.cuda.device_count()
    server = AsyncGenerationServer(model_class(), host, port, num_devices)
    server.run()

if __name__ == "__main__":
    app()