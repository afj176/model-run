"""ComfyUI custom-node entry point; install this repository in custom_nodes."""
from .model_run.comfy_bridge import ModelRunObserve, install_routes

NODE_CLASS_MAPPINGS = {"ModelRunObserve": ModelRunObserve}
NODE_DISPLAY_NAME_MAPPINGS = {"ModelRunObserve": "Model Run · Observe"}
WEB_DIRECTORY = "./comfy_web"

install_routes()
