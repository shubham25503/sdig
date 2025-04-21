from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import StreamingResponse
from typing import List
from app.services.image_generator2 import generate_images
from PIL import Image
from io import BytesIO
import zipfile
import os 
import base64
app = FastAPI()

@app.post("/generate/")
async def generate_images_api(
    injection_number: int = Form(...),
    selected_area: str = Form(...),
    file: UploadFile = File(...)
):
    try:
        image = Image.open(BytesIO(await file.read())).convert("RGB")
        image = resize_image_dynamic(image)
        result_image = generate_images(image, selected_area, injection_number)[0]

        if not result_image:
            return {"error": "No generated image found"}

        result_image_bytes = result_image["image_bytes"]

        # Base64 encode the image bytes
        encoded_image = base64.b64encode(result_image_bytes).decode('utf-8')

        return {"image": encoded_image}
    except Exception as e:
        print(e)
        return e

from PIL import Image

def resize_image_dynamic(image: Image.Image, min_size=512, max_size=1024):
    width, height = image.size

    # Find the longer side and scale factor
    max_side = max(width, height)
    min_side = min(width, height)

    # If already within desired range, skip resizing
    if min_size <= max_side <= max_size:
        return image

    # Compute new size while preserving aspect ratio
    if max_side > max_size:
        scale_factor = max_size / max_side
    else:
        scale_factor = min_size / max_side

    new_width = int(width * scale_factor)
    new_height = int(height * scale_factor)

    resized_image = image.resize((new_width, new_height), Image.LANCZOS)
    return resized_image


    # Create filename like originalname-inj3.jpg
    # original_filename = os.path.splitext(file.filename)[0]
    # extension = os.path.splitext(file.filename)[1] or ".jpg"
    # new_filename = f"{original_filename}-inj{injection_number}{extension}"

    # Prepare image for streaming
    # buffer = BytesIO(result["image_bytes"])
    # buffer.seek(0)

    # return StreamingResponse(buffer, media_type="image/jpeg", headers={
    #     "Content-Disposition": f"attachment; filename={new_filename}"
    # })
