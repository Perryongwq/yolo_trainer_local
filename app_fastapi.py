from fastapi import FastAPI, Request, HTTPException, File, UploadFile, Form
from fastapi.responses import (
    JSONResponse,
    StreamingResponse,
    FileResponse,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
from pydantic import BaseModel
from typing import Optional, Dict, Any, AsyncGenerator
from contextlib import asynccontextmanager
import os
import sys
import cv2
import pandas as pd
import base64
from datetime import datetime
from pathlib import Path
from ultralytics import YOLO
import threading
import time
import numpy as np
import logging
import uvicorn
import torch
import asyncio

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Paths - Handle both development and frozen executable modes
# IMPORTANT: Determine absolute paths at module load time, never rely on cwd() later
if getattr(sys, 'frozen', False):
    # Running as frozen executable
    # The launcher sets cwd to backend data dir BEFORE importing this module
    # We capture it immediately and use absolute paths from here on
    BASE_DATA_DIR = Path(sys.executable).parent / "backend"
    # Ensure the directory exists
    BASE_DATA_DIR.mkdir(parents=True, exist_ok=True)
else:
    # Development mode - use the backend directory
    BASE_DATA_DIR = Path(__file__).parent

# Convert to absolute paths - these will never change even if cwd() changes later
UPLOAD_FOLDER = str(BASE_DATA_DIR / "uploads")
PROCESSED_FOLDER = str(BASE_DATA_DIR / "processed")
RESULTS_FOLDER = str(BASE_DATA_DIR / "results")
EXCEL_FILE = str(BASE_DATA_DIR / "processed" / "measurement_results.xlsx")

# Create folders at startup
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

logger.info(f"=== Backend Data Directories Initialized ===")
logger.info(f"BASE_DATA_DIR: {BASE_DATA_DIR}")
logger.info(f"UPLOAD_FOLDER: {UPLOAD_FOLDER}")
logger.info(f"PROCESSED_FOLDER: {PROCESSED_FOLDER}")
logger.info(f"RESULTS_FOLDER: {RESULTS_FOLDER}")
logger.info(f"Frozen mode: {getattr(sys, 'frozen', False)}")
logger.info(f"===========================================")


# Global storage for latest processing results
latest_processing_results = {
    "y_diff_microns": None,
    "judgment": None,
    "processed_timestamp": None,
}


# Lifespan context manager for startup and shutdown
@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator:
    """Manage application lifecycle"""
    # Startup
    logger.info("Application starting up...")
    yield
    # Shutdown
    global camera
    logger.info("Application shutting down...")
    if camera is not None and camera.isOpened():
        camera.release()
        camera = None
        logger.info("Camera released on shutdown")


# Initialize FastAPI app as API-only backend
app = FastAPI(
    title="CT600 Vision Guide API",
    version="1.0.0",
    description="Backend API for CT600 Vision Inspection System",
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Custom middleware to handle large file uploads
@app.middleware("http")
async def process_file_upload(request: Request, call_next):
    # Log request size if available
    content_length = request.headers.get("content-length")
    if content_length:
        size_mb = int(content_length) / (1024 * 1024)
        logger.info(f"Request size: {content_length} bytes ({size_mb:.2f} MB)")

    # Set a more reasonable limit (50MB)
    max_size = 50 * 1024 * 1024  # 50MB

    if content_length and int(content_length) > max_size:
        logger.warning(f"Request exceeds size limit: {size_mb:.2f} MB")
        return JSONResponse(
            status_code=413,
            content={
                "success": False,
                "message": f"Request too large: {size_mb:.2f} MB. Maximum allowed: 50 MB",
            },
        )

    # Continue with the request
    response = await call_next(request)
    return response


# Your existing middleware for request logging
@app.middleware("http")
async def log_requests(request: Request, call_next):
    logger.info(f"Request method: {request.method}")
    logger.info(f"Request URL: {request.url}")
    logger.info(
        f"Content-Type header: {request.headers.get('Content-Type', 'Not set')}"
    )

    # Continue with the request
    response = await call_next(request)
    return response


# Note: Static files and templates removed - frontend handles all UI

# Load YOLO model for 15type only - dynamically load from models directory
# Get the directory where this script is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
MODEL_FILENAME = "15type_model.pt"
MODEL_PATH = os.path.join(MODELS_DIR, MODEL_FILENAME)

# Validate model file exists
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Model file not found at {MODEL_PATH}. Please ensure the model file exists."
    )

logger.info(f"Loading YOLO model from: {MODEL_PATH}")
model = YOLO(MODEL_PATH)
class_names_15type = [
    "block1_edge15",
    "block2_edge15",
    "block1_15",
    "block2_15",
    "cal_mark",
]

# Constants
MICRONS_PER_PIXEL = 2.3
BLOCK1_OFFSET = 0.0
BLOCK2_OFFSET = 0.0
MEASUREMENT_OFFSET_MICRONS = (
    5.0  # Offset to apply to final measurement (e.g., +5 to correct camera calibration)
)
judgment_criteria = {"good": 10, "acceptable": 20}

# Camera setup - Initialize as None for lazy loading
camera = None
lock = threading.Lock()
frame = None


# Helper function to initialize camera with optimized settings
def init_camera_optimized():
    """Initialize camera with DirectShow backend for faster performance on Windows"""
    cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # DirectShow backend for Windows
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    cam.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer for lower latency
    cam.set(cv2.CAP_PROP_FPS, 30)  # Set target FPS
    return cam


# Pydantic models for request/response
class ImageSubmission(BaseModel):
    image_url: str
    machine_number: str


class ManualSubmission(BaseModel):
    aligned_image: str
    manual_lines: list


class AlignedImageSubmission(BaseModel):
    aligned_image: str


# Exception handlers
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    logger.error(f"Validation error: {exc}")
    return JSONResponse(
        status_code=400,
        content={
            "success": False,
            "message": "Bad request. Please check your data format.",
            "error_code": 400,
            "details": str(exc),
        },
    )


@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    if exc.status_code == 413:
        logger.error(f"413 error: Request entity too large")
        return JSONResponse(
            status_code=413,
            content={
                "success": False,
                "message": "File too large. Maximum size is 50MB. Please compress your image or reduce quality.",
                "error_code": 413,
            },
        )
    logger.error(f"HTTP error {exc.status_code}: {exc.detail}")
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "message": str(exc.detail),
            "error_code": exc.status_code,
        },
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"success": False, "message": f"Server error: {str(exc)}"},
    )


# Middleware for request logging
@app.middleware("http")
async def log_requests(request: Request, call_next):
    logger.info(f"Request method: {request.method}")
    logger.info(f"Request URL: {request.url}")
    logger.info(
        f"Content-Type header: {request.headers.get('Content-Type', 'Not set')}"
    )

    # Check content length if available
    content_length = request.headers.get("content-length")
    if content_length:
        size_mb = int(content_length) / (1024 * 1024)
        logger.info(f"Request size: {content_length} bytes ({size_mb:.2f} MB)")

        # Early validation for large requests
        if int(content_length) > 100 * 1024 * 1024:  # 100MB limit
            logger.warning(f"Request exceeds size limit: {size_mb:.2f} MB")
            return JSONResponse(
                status_code=413,
                content={
                    "success": False,
                    "message": f"Request too large: {size_mb:.2f} MB. Maximum allowed: 100 MB",
                },
            )

    response = await call_next(request)
    return response


@app.get("/video_feed")
async def video_feed():
    def generate_frames():
        global frame, camera
        while True:
            # Check if camera is available
            if camera is None or not camera.isOpened():
                logger.warning("Camera not available in video_feed, reinitializing...")
                camera = init_camera_optimized()
                if not camera.isOpened():
                    logger.error("Failed to reinitialize camera in video_feed")
                    break

            ret, frame = camera.read()
            if not ret:
                logger.warning("Failed to read frame from camera")
                break

            # Encode with lower quality for faster streaming
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 75]
            _, buffer = cv2.imencode(".jpg", frame, encode_param)
            frame_bytes = buffer.tobytes()
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
            )

    return StreamingResponse(
        generate_frames(), media_type="multipart/x-mixed-replace; boundary=frame"
    )


@app.post("/capture")
async def capture_image():
    global frame, camera
    logger.info("Capture image endpoint called")

    if camera is None or not camera.isOpened():
        camera = init_camera_optimized()
        if not camera.isOpened():
            logger.error("Unable to access camera")
            raise HTTPException(status_code=500, detail="Unable to access the camera.")

    with lock:
        ret, frame = camera.read()
        if not ret:
            logger.error("Failed to capture frame")
            raise HTTPException(status_code=500, detail="Failed to capture image.")

        captured_path = os.path.join(UPLOAD_FOLDER, "captured_image.png")
        success = cv2.imwrite(captured_path, frame)

        if not success or not os.path.exists(captured_path):
            logger.error("Failed to write captured image to disk")
            raise HTTPException(
                status_code=500, detail="Failed to save captured image."
            )

        # Verify file size
        file_size = os.path.getsize(captured_path)
        logger.info(
            f"Image captured and saved to {captured_path}, size: {file_size} bytes"
        )

        if file_size == 0:
            logger.error("Captured image is empty")
            raise HTTPException(status_code=500, detail="Captured image is empty.")

    # Small delay to ensure file is completely written
    await asyncio.sleep(0.05)  # Reduced from 0.1 to 0.05 for faster response

    return {
        "success": True,
        "image_path": f"/uploads/captured_image.png",
        "proceed_to_crop": True,
    }


@app.post("/reconnect_camera")
async def reconnect_camera():
    global camera
    logger.info("Reconnecting camera")

    # Run camera operations in thread pool to avoid blocking
    def _reconnect():
        global camera

        # Release existing camera if open (with timeout)
        if camera is not None:
            try:
                if camera.isOpened():
                    camera.release()
                    logger.info("Released existing camera connection")
            except Exception as e:
                logger.warning(f"Error releasing camera (ignoring): {e}")
            camera = None

        # Small delay to ensure camera is fully released
        time.sleep(0.05)

        # Initialize new camera connection with optimized settings
        new_camera = init_camera_optimized()

        if new_camera.isOpened():
            logger.info("Camera reconnected successfully")
            return new_camera
        else:
            logger.error("Failed to open camera")
            return None

    # Execute in thread pool
    loop = asyncio.get_event_loop()
    new_cam = await loop.run_in_executor(None, _reconnect)

    if new_cam is None:
        logger.error("Failed to reconnect camera")
        camera = None
        raise HTTPException(status_code=500, detail="Failed to reconnect camera.")

    camera = new_cam
    return {"success": True, "message": "Camera reconnected successfully!"}


@app.post("/disconnect_camera")
async def disconnect_camera():
    global camera
    logger.info("Disconnecting camera")

    # Run camera release in thread pool for faster response
    def _disconnect():
        global camera
        if camera is not None:
            try:
                if camera.isOpened():
                    camera.release()
                    logger.info("Camera disconnected successfully")
            except Exception as e:
                logger.warning(f"Error disconnecting camera (ignoring): {e}")
            camera = None
            return True
        else:
            logger.info("Camera already disconnected")
            camera = None
            return False

    loop = asyncio.get_event_loop()
    was_open = await loop.run_in_executor(None, _disconnect)

    if was_open:
        return {"success": True, "message": "Camera disconnected successfully!"}
    else:
        return {"success": True, "message": "Camera was already disconnected."}


@app.get("/uploads/{filename}")
async def uploaded_file(filename: str):
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    if not os.path.exists(file_path):
        logger.error(f"File not found: {file_path}")
        raise HTTPException(status_code=404, detail="File not found")

    logger.info(f"Serving file: {file_path}, size: {os.path.getsize(file_path)} bytes")

    return FileResponse(
        file_path,
        media_type="image/png",
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache",
            "Expires": "0",
            "Access-Control-Allow-Origin": "*",
        },
    )


@app.get("/processed/{filename}")
async def processed_file(filename: str):
    file_path = os.path.join(PROCESSED_FOLDER, filename)
    if not os.path.exists(file_path):
        logger.error(f"Processed file not found: {file_path}")
        raise HTTPException(status_code=404, detail="File not found")

    logger.info(
        f"Serving processed file: {file_path}, size: {os.path.getsize(file_path)} bytes"
    )

    return FileResponse(
        file_path,
        media_type="image/png",
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache",
            "Expires": "0",
            "Access-Control-Allow-Origin": "*",
        },
    )


@app.get("/results/{machine_number}/{filename}")
async def results_file(machine_number: str, filename: str):
    """Serve saved result images from results folder"""
    file_path = os.path.join(RESULTS_FOLDER, machine_number, filename)
    if not os.path.exists(file_path):
        logger.error(f"Result file not found: {file_path}")
        raise HTTPException(status_code=404, detail="File not found")

    logger.info(
        f"Serving result file: {file_path}, size: {os.path.getsize(file_path)} bytes"
    )

    return FileResponse(
        file_path,
        media_type="image/png",
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache",
            "Expires": "0",
            "Access-Control-Allow-Origin": "*",
        },
    )


@app.post("/save-image")
async def save_image(request: Request):
    logger.info(f"Save-image route called")

    try:
        data = await request.json()
        if not data:
            logger.error("No JSON data received")
            raise HTTPException(status_code=400, detail="No data received.")

        image_url = data.get("image_url")
        machine_number = data.get("machine_number", "").strip().lower()
        username = data.get("username", "Unknown")  # Get username from request
        human_judgement = data.get("human_judgement", "")  # Get human judgement from request

        # Validate machine number format
        if (
            not machine_number.startswith("ct")
            or not machine_number[2:].isdigit()
            or len(machine_number) != 5
        ):
            logger.error(f"Invalid machine number format: {machine_number}")
            raise HTTPException(
                status_code=400,
                detail="Invalid machine number format. Use 'ct' followed by 3 digits.",
            )

        if not image_url:
            logger.error("Image URL not provided")
            raise HTTPException(status_code=400, detail="Image URL not provided.")

        # Decode base64 image with size validation
        try:
            if "," in image_url:
                header, encoded = image_url.split(",", 1)
            else:
                encoded = image_url

            # Check base64 size before decoding
            encoded_size_mb = (
                len(encoded) * 3 / 4 / (1024 * 1024)
            )  # Approximate decoded size
            logger.info(f"Base64 encoded size: ~{encoded_size_mb:.2f} MB")

            if encoded_size_mb > 40:  # Leave some buffer from 50MB limit
                raise HTTPException(
                    status_code=413,
                    detail=f"Image too large: ~{encoded_size_mb:.2f} MB. Please compress the image.",
                )

            decoded_image = base64.b64decode(encoded)
            logger.info(f"Image decoded successfully, size: {len(decoded_image)} bytes")

        except Exception as e:
            logger.error(f"Image decoding failed: {e}")
            raise HTTPException(
                status_code=400, detail=f"Image decoding failed: {str(e)}"
            )

        # Save to results folder in backend directory
        result_folder = os.path.join(RESULTS_FOLDER, machine_number)
        os.makedirs(result_folder, exist_ok=True)

        # Generate timestamped filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        image_filename = f"{machine_number}_{timestamp}.png"
        image_path = os.path.join(result_folder, image_filename)

        # Save image
        with open(image_path, "wb") as f:
            f.write(decoded_image)

        logger.info(f"Image saved to: {image_path}")

        # Get the latest processing results
        y_diff = latest_processing_results.get("y_diff_microns", "N/A")
        judgment = latest_processing_results.get("judgment", "N/A")

        # Prepare new row with all required information
        new_row = {
            "Username": username,
            "Image Name": image_filename,
            "Y-Difference (microns)": y_diff,
            "AI Judgment": judgment,
            "Checked Date and Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Machine Number": machine_number.upper(),
            "Human Judgement": human_judgement,
            "Saved Path": image_path,
        }

        # Create machine-specific Excel file
        excel_filename = f"{machine_number}_results.xlsx"
        excel_path = os.path.join(result_folder, excel_filename)

        try:
            if os.path.exists(excel_path):
                # Append to existing Excel file
                df_existing = pd.read_excel(excel_path)
                df_updated = pd.concat(
                    [df_existing, pd.DataFrame([new_row])], ignore_index=True
                )
                logger.info(f"Appending to existing Excel file: {excel_path}")
            else:
                # Create new Excel file
                df_updated = pd.DataFrame([new_row])
                logger.info(f"Creating new Excel file: {excel_path}")

            df_updated.to_excel(excel_path, index=False)
            logger.info(f"Excel file updated successfully: {excel_path}")

        except Exception as e:
            logger.error(f"Failed to update Excel file: {e}")
            raise HTTPException(
                status_code=500, detail=f"Failed to save to Excel: {str(e)}"
            )

        return {
            "success": True,
            "message": "Image and result saved!",
            "excel_path": excel_path,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in save_image: {e}")
        raise HTTPException(status_code=500, detail=f"Server error: {str(e)}")


@app.post("/manual-submit")
async def manual_submit(request: Request):
    logger.info(f"Manual-submit route called")

    try:
        data = await request.json()
        if not data:
            logger.error("No JSON data received in manual-submit")
            raise HTTPException(status_code=400, detail="No data received.")

        aligned_image_data = data.get("aligned_image")
        manual_lines = data.get("manual_lines")

        if not aligned_image_data or not manual_lines or len(manual_lines) != 2:
            logger.error("Missing or invalid data in manual-submit")
            raise HTTPException(status_code=400, detail="Missing or invalid data.")

        # Decode image with size validation
        try:
            if "," in aligned_image_data:
                header, encoded = aligned_image_data.split(",", 1)
            else:
                encoded = aligned_image_data

            # Check size before decoding
            encoded_size_mb = len(encoded) * 3 / 4 / (1024 * 1024)
            logger.info(f"Manual submit image size: ~{encoded_size_mb:.2f} MB")

            image_bytes = base64.b64decode(encoded)
            nparr = np.frombuffer(image_bytes, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if image is None:
                raise ValueError("Failed to decode image")

        except Exception as e:
            logger.error(f"Image processing failed in manual-submit: {e}")
            raise HTTPException(
                status_code=400, detail=f"Image processing failed: {str(e)}"
            )

        microns_per_pixel = MICRONS_PER_PIXEL
        h, w = image.shape[:2]
        y1 = int(manual_lines[0]["y1"])  # block1_edge y
        y2 = int(manual_lines[1]["y1"])  # block2_edge y
        y1 = max(0, min(h - 1, y1))
        y2 = max(0, min(h - 1, y2))

        # Draw straight (non-slanted) lines across the image width
        cv2.line(image, (0, y1), (w - 1, y1), (255, 0, 0), 2)  # block1_edge
        cv2.line(image, (0, y2), (w - 1, y2), (0, 255, 255), 2)  # block2_edge

        # Signed difference: block1_edge - block2_edge (no abs here)
        y_diff_pixels = y1 - y2
        y_diff_microns = (
            y_diff_pixels * microns_per_pixel
        ) + MEASUREMENT_OFFSET_MICRONS

        current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Keep judgment on magnitude (unchanged logic)
        if y_diff_microns < 0:
            judgment = "Good"
            judgment_color = (0, 255, 0)
        elif 0 <= y_diff_microns <= 10:
            judgment = "Acceptable"
            judgment_color = (0, 165, 255)
        else:
            judgment = "No Good"
            judgment_color = (0, 0, 255)

        # Annotations
        mid_y = int((y1 + y2) / 2)
        text_x = w // 2 + 250  # Shifted to the right
        cv2.putText(
            image,
            f"{y_diff_microns:.2f} microns (b1-b2)",
            (text_x - 50, mid_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
        )
        cv2.putText(
            image,
            f"Judgment: {judgment}",
            (text_x - 50, mid_y + 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            judgment_color,
            2,
        )
        cv2.putText(
            image,
            f"Checked on: {current_datetime}",
            (10, h - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 255, 255),
            3,
        )
        cv2.putText(
            image,
            "Manual Judgment",
            (text_x - 120, mid_y + 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 255),
            2,
        )
        cv2.putText(
            image,
            f"{microns_per_pixel:.2f} um/pixel",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 0),
            2,
        )

        processed_path = os.path.join(PROCESSED_FOLDER, "processed_image.png")
        cv2.imwrite(processed_path, image)
        logger.info(f"Processed image saved to: {processed_path}")

        # Store results in global variable for later use when saving
        global latest_processing_results
        latest_processing_results = {
            "y_diff_microns": round(y_diff_microns, 2),
            "judgment": judgment,
            "processed_timestamp": current_datetime,
        }
        logger.info(f"Stored processing results: {latest_processing_results}")

        # Save to Excel
        try:
            df = pd.DataFrame(
                [
                    {
                        "Image Name": "manual_drawn_image.png",
                        "Y-Difference (microns)": y_diff_microns,
                        "Judgment": judgment,
                        "Checked Date and Time": current_datetime,
                    }
                ]
            )
            df.to_excel(EXCEL_FILE, index=False)
            logger.info("Manual judgment saved to Excel")
        except Exception as e:
            logger.error(f"Failed to save manual judgment to Excel: {e}")

        return {"processed_image_url": f"/processed/processed_image.png"}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in manual_submit: {e}")
        raise HTTPException(status_code=500, detail=f"Server error: {str(e)}")


@app.post("/test-upload")
async def test_upload(request: Request):
    """Test endpoint to debug upload issues"""
    logger.info(f"Test upload called")

    try:
        content_type = request.headers.get("content-type", "")
        if "application/json" in content_type:
            data = await request.json()
            image_data = data.get("aligned_image", "")
            size_mb = len(image_data) * 3 / 4 / (1024 * 1024)
            logger.info(f"JSON data received, image size: ~{size_mb:.2f} MB")
            return {
                "success": True,
                "message": f"JSON upload test successful, size: {size_mb:.2f} MB",
            }
        else:
            form_data = await request.form()
            form_image = form_data.get("aligned_image", "")
            size_mb = len(str(form_image)) * 3 / 4 / (1024 * 1024)
            logger.info(f"Form data received, image size: ~{size_mb:.2f} MB")
            return {
                "success": True,
                "message": f"Form upload test successful, size: {size_mb:.2f} MB",
            }
    except Exception as e:
        logger.error(f"Test upload failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
async def root():
    """Root endpoint - API information"""
    return {
        "service": "CT600 Vision Guide API",
        "version": "1.0.0",
        "status": "running",
        "frontend_url": "http://localhost:5001/vision-inspection",
        "endpoints": {
            "health": "/health",
            "video_feed": "/video_feed",
            "capture": "/capture",
            "reconnect_camera": "/reconnect_camera",
            "process_image": "/",
            "manual_submit": "/manual-submit",
            "save_image": "/save-image",
        },
    }


@app.get("/health")
async def health_check():
    """Health check endpoint for monitoring"""
    return {
        "status": "healthy",
        "service": "CT600 Vision API",
        "timestamp": datetime.now().isoformat(),
    }


@app.post("/")
async def index_post(request: Request):
    logger.info(f"Index POST route called")

    try:
        # Clear previous files
        for folder in [UPLOAD_FOLDER, PROCESSED_FOLDER]:
            for filename in os.listdir(folder):
                file_path = os.path.join(folder, filename)
                if os.path.isfile(file_path):
                    os.unlink(file_path)

        # Try to get the aligned_image from JSON or form data
        aligned_image_data = None
        try:
            # First try JSON data
            content_type = request.headers.get("content-type", "")
            if "application/json" in content_type:
                json_data = await request.json()
                if json_data:
                    aligned_image_data = json_data.get("aligned_image")
                    logger.info("Successfully retrieved aligned_image from JSON data")
            else:
                # Fallback to form data
                form_data = await request.form()
                aligned_image_data = form_data.get("aligned_image")
                logger.info("Successfully retrieved aligned_image from form data")
        except Exception as parse_error:
            logger.error(f"Failed to parse request data: {parse_error}")
            # If both fail, try to get raw data
            try:
                raw_data = await request.body()
                raw_text = raw_data.decode("utf-8")
                logger.info(f"Retrieved raw data, length: {len(raw_text)}")
                # Try to extract base64 data from raw form data
                if "data:image" in raw_text:
                    import re

                    base64_match = re.search(
                        r"data:image/[^;]+;base64,([A-Za-z0-9+/=]+)", raw_text
                    )
                    if base64_match:
                        aligned_image_data = (
                            f"data:image/png;base64,{base64_match.group(1)}"
                        )
                        logger.info("Extracted base64 data from raw data")
            except Exception as raw_error:
                logger.error(f"Failed to parse raw data: {raw_error}")

        if not aligned_image_data:
            logger.error("No image data provided in index POST")
            raise HTTPException(status_code=400, detail="No image data provided.")

        # Decode and validate image
        try:
            if "," in aligned_image_data:
                header, encoded = aligned_image_data.split(",", 1)
            else:
                encoded = aligned_image_data

            # Size check
            encoded_size_mb = len(encoded) * 3 / 4 / (1024 * 1024)
            logger.info(f"Processing image size: ~{encoded_size_mb:.2f} MB")

            decoded_image = base64.b64decode(encoded)
            nparr = np.frombuffer(decoded_image, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if image is None:
                raise ValueError("Failed to decode image")

        except Exception as e:
            logger.error(f"Failed to decode image in index: {e}")
            raise HTTPException(
                status_code=400, detail=f"Failed to decode image: {str(e)}"
            )

        # YOLO prediction
        results = model.predict(source=image, conf=0.25, save=False)

        block1_edge_y = block2_edge_y = None
        block1_box_y = block2_box_y = None
        calibration_marker_width_px = None
        microns_per_pixel = MICRONS_PER_PIXEL

        for box, cls in zip(results[0].boxes.xywh, results[0].boxes.cls):
            x_center, y_center, width, height = box
            label = class_names_15type[int(cls.item())]
            logger.info(f"[DEBUG] cls: {cls}, index: {int(cls.item())}, label: {label}")

            if label == "block1_edge15":
                edge_y = int(y_center + height / 2)
                block1_edge_y = edge_y + (BLOCK1_OFFSET / microns_per_pixel)
                cv2.line(
                    image,
                    (int(x_center - 150), edge_y),
                    (int(x_center + 150), edge_y),
                    (255, 0, 0),
                    2,
                )

            elif label == "block2_edge15":
                edge_y = int(y_center + height / 2)
                block2_edge_y = edge_y + (BLOCK2_OFFSET / microns_per_pixel)
                cv2.line(
                    image,
                    (int(x_center - 150), edge_y),
                    (int(x_center + 150), edge_y),
                    (0, 255, 255),
                    2,
                )

            elif label == "block1_15":
                block1_box_y = int(y_center + height / 2)

            elif label == "block2_15":
                block2_box_y = int(y_center + height / 2)

            elif label == "cal_mark":
                calibration_marker_width_px = width.item()

        # Calibration check
        if calibration_marker_width_px:
            microns_per_pixel = 1000.0 / calibration_marker_width_px
            logger.info(
                f"[Calibration] cal_mark width = {calibration_marker_width_px:.2f}px, microns/px = {microns_per_pixel:.2f}"
            )
            print(
                f"[Calibration] cal_mark width = {calibration_marker_width_px:.2f}px, microns/px = {microns_per_pixel:.2f}"
            )

            if microns_per_pixel > 10:
                logger.warning(
                    "Microns per pixel too high, suggesting focus adjustment"
                )
                return {
                    "show_fallback_modal": True,
                    "reason": "Please fine tune the focus and take again.",
                }
        else:
            logger.warning("cal_mark not detected, using manual mode")
            return {
                "show_manual_draw_modal": True,
                "reason": "cal_mark is not available. Draw the manual line and judge.",
            }

        # Check if both edge positions are available
        if block1_edge_y is None or block2_edge_y is None:
            logger.warning("One or more edges not detected")
            return {
                "show_fallback_modal": True,
                "reason": "Not able to detect one or more edges. Please draw manual lines and submit.",
            }

        # Calculate measurements
        y_diff_pixels = block1_edge_y - block2_edge_y
        y_diff_microns = (
            y_diff_pixels * microns_per_pixel
        ) + MEASUREMENT_OFFSET_MICRONS
        current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Judgment logic
        if y_diff_microns < judgment_criteria["good"]:
            judgment = "Good"
            judgment_color = (0, 255, 0)
        elif y_diff_microns < judgment_criteria["acceptable"]:
            judgment = "Acceptable"
            judgment_color = (0, 165, 255)
        else:
            judgment = "No Good"
            judgment_color = (0, 0, 255)

        # Add annotations
        text_x = image.shape[1] // 2 + 250  # Shifted to the right
        text_y = int((block1_edge_y + block2_edge_y) / 2)
        cv2.putText(
            image,
            f"{y_diff_microns:.2f} microns",
            (text_x - 100, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
        )
        cv2.putText(
            image,
            f"Judgment: {judgment}",
            (text_x - 100, text_y + 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            judgment_color,
            2,
        )
        cv2.putText(
            image,
            f"Checked on: {current_datetime}",
            (10, image.shape[0] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 255, 255),
            3,
        )

        # Save processed image
        processed_path = os.path.join(PROCESSED_FOLDER, "processed_image.png")
        cv2.imwrite(processed_path, image)
        logger.info(f"Processed image saved to: {processed_path}")

        # Store results in global variable for later use when saving
        global latest_processing_results
        latest_processing_results = {
            "y_diff_microns": round(y_diff_microns, 2),
            "judgment": judgment,
            "processed_timestamp": current_datetime,
        }
        logger.info(f"Stored processing results: {latest_processing_results}")

        # Save results to Excel
        try:
            results_df = pd.DataFrame(
                [
                    {
                        "Image Name": "aligned_image.png",
                        "Y-Difference (microns)": y_diff_microns,
                        "Judgment": judgment,
                        "Checked Date and Time": current_datetime,
                    }
                ]
            )
            results_df.to_excel(EXCEL_FILE, index=False)
            logger.info("Results saved to Excel")
        except Exception as e:
            logger.error(f"Failed to save results to Excel: {e}")

        return {
            "processed_image_url": f"/processed/processed_image.png",
            "show_final_confirm_popup": True,
            "final_message": "Please confirm visually for guide position.",
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in index POST: {e}")
        raise HTTPException(status_code=500, detail=f"Server error: {str(e)}")


if __name__ == "__main__":
    logger.info("Starting FastAPI application...")
    uvicorn.run(
        "app_fastapi:app", host="localhost", port=5000, log_level="info", reload=True
    )
