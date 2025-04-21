from diffusers import StableDiffusionImg2ImgPipeline
import torch

def load_model():
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float16
    ).to("cuda")
    pipe.enable_model_cpu_offload()
    pipe.safety_checker = None
    return pipe

pipeline = load_model()
