import os
import torch
from fastapi import FastAPI, File, UploadFile, Form, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Dict
import cv2
import numpy as np
import mediapipe as mp
import json
import asyncio
from services.image_generator2 import generate_images
from PIL import Image
from io import BytesIO
import zipfile
import base64

# Set the CUDA environment variable to manage memory fragmentation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"  # Adjust split size as needed

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Check if CUDA is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Initialize MediaPipe with GPU support
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Injection points configuration
INJECTION_POINTS = {
    "forehead_lines_botox": [10, 151, 338, 67, 107, 336],
    "frown_lines_glabella_botox": [9, 8, 168, 6, 197, 195, 5],
    "crows_feet_botox": [263, 362, 386, 133, 173, 156],
    "nasalis_lines_botox": [98, 327],
    "vertical_lip_lines_botox": [61, 0, 267, 17, 84, 37],
    "lip_flip_botox": [0, 267, 37, 17, 287, 84],
    "smile_lift_botox": [61, 76, 91, 305, 290, 409],
    "masseter_reduction_botox": [172, 58],
    "dimpled_chin_botox": [18, 83, 14],
    "platysmal_bands_botox": [131, 50, 205, 280, 425],
    "cheek_filler": [50, 205, 429, 280, 425, 449],
    "smile_line_filler": [61, 91, 76, 290, 305, 409],
    "lip_filler": [0, 267, 37, 17, 287, 84],
    "temple_filler": [26, 54, 226, 247, 110],
    "nose_filler": [4, 5, 6, 248]
}

# Cache for smoothing landmark positions
landmark_cache: Dict[str, tuple] = {}

# Store active connections
active_connections: List[WebSocket] = []

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    active_connections.append(websocket)
    
    try:
        while True:
            # Receive binary data (image)
            data = await websocket.receive_bytes()
            
            # Convert bytes to numpy array
            nparr = np.frombuffer(data, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            # Process frame with MediaPipe
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = face_mesh.process(frame_rgb)
            
            landmark_pixel_list = []
            
            if result.multi_face_landmarks:
                for face_landmarks in result.multi_face_landmarks:
                    h, w, _ = frame.shape
                    for name, indices in INJECTION_POINTS.items():
                        for idx in indices:
                            if idx < len(face_landmarks.landmark):
                                lm = face_landmarks.landmark[idx]
                                x, y = int(lm.x * w), int(lm.y * h)
                                
                                # Smooth using cache
                                prev = landmark_cache.get(f"{name}_{idx}")
                                if prev:
                                    x = int(prev[0] * 0.7 + x * 0.3)
                                    y = int(prev[1] * 0.7 + y * 0.3)
                                
                                landmark_cache[f"{name}_{idx}"] = (x, y)
                                landmark_pixel_list.append({
                                    "name": name,
                                    "index": idx,
                                    "x": x,
                                    "y": y
                                })
                                
                                # Draw point
                                cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)
            
            # Send landmarks as JSON
            await websocket.send_json({"landmarks": landmark_pixel_list})
            
            # Send processed frame as binary
            _, buffer = cv2.imencode('.jpg', frame)
            await websocket.send_bytes(buffer.tobytes())
            
    except WebSocketDisconnect:
        active_connections.remove(websocket)
    except Exception as e:
        print(f"Error in websocket_endpoint: {str(e)}")
        if websocket in active_connections:
            active_connections.remove(websocket)

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
