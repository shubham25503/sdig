from diffusers import StableDiffusionImg2ImgPipeline
import torch

def load_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading model on: {device}")

    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float16 if device == "cuda" else torch.float32
    ).to(device)

    if device == "cuda":
        pipe.enable_attention_slicing()
        pipe.enable_model_cpu_offload()

    pipe.safety_checker = None
    return pipe

pipeline = load_model()
