import os
import torch
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import StreamingResponse
from typing import List
from services.image_generator2 import generate_images
from PIL import Image
from io import BytesIO
import zipfile
import base64

# Set the CUDA environment variable to manage memory fragmentation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"  # Adjust split size as needed

app = FastAPI()

@app.post("/generate/")
async def generate_images_api(
    injection_number: int = Form(...),
    selected_area: str = Form(...),
    file: UploadFile = File(...)
):
    try:
        # Load image from the uploaded file
        image = Image.open(BytesIO(await file.read())).convert("RGB")
        
        # Monitor initial memory usage
        print(f"Initial Allocated Memory: {torch.cuda.memory_allocated()} bytes")
        print(f"Initial Reserved Memory: {torch.cuda.memory_reserved()} bytes")
        
        # Generate the image with the provided function
        result_image = generate_images(image, selected_area, injection_number)[0]

        # If no image is returned, raise an error
        if not result_image:
            return {"error": "No generated image found"}

        result_image_bytes = result_image["image_bytes"]

        # Base64 encode the image bytes for returning as a response
        encoded_image = base64.b64encode(result_image_bytes).decode('utf-8')

        # Clean up and clear the GPU cache
        torch.cuda.empty_cache()
        
        # Monitor memory usage after generation and cleanup
        print(f"Allocated Memory after: {torch.cuda.memory_allocated()} bytes")
        print(f"Reserved Memory after: {torch.cuda.memory_reserved()} bytes")

        return {"image": encoded_image}

    except Exception as e:
        print(f"Error during image generation: {e}")
        return {"error": str(e)}
