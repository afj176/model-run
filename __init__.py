"""ComfyUI custom-node entry point; install this repository in custom_nodes."""
from .model_run.comfy_bridge import ModelRunAttachOutput, ModelRunObserve, install_routes
from .model_run.comfy_components import ModelRunTextEncode, ModelRunTextGenerate, ModelRunVAEDecode, ModelRunVAEEncode

NODE_CLASS_MAPPINGS = {"ModelRunObserve": ModelRunObserve, "ModelRunAttachOutput": ModelRunAttachOutput, "ModelRunTextEncode": ModelRunTextEncode, "ModelRunTextGenerate": ModelRunTextGenerate,
                       "ModelRunVAEDecode": ModelRunVAEDecode, "ModelRunVAEEncode": ModelRunVAEEncode}
NODE_DISPLAY_NAME_MAPPINGS = {"ModelRunObserve": "Model Run · Observe", "ModelRunAttachOutput": "Model Run · Attach Output",
                              "ModelRunTextEncode": "Model Run · Text Encode", "ModelRunTextGenerate": "Model Run · Generate Text", "ModelRunVAEDecode": "Model Run · VAE Decode", "ModelRunVAEEncode": "Model Run · VAE Encode"}
WEB_DIRECTORY = "./comfy_web"

install_routes()
