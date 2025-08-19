from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
from transformers import AutoProcessor, AutoModel
from PIL import Image
import io
import base64
import numpy as np
from typing import List, Union
import logging
import os
import asyncio

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="SigLIP Embedding API",
    description="API for generating text and image embeddings using SigLIP model",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for model and processor
model = None
processor = None
device = None
model_loaded = False

class TextRequest(BaseModel):
    text: Union[str, List[str]]

class TextEmbeddingResponse(BaseModel):
    embeddings: List[List[float]]
    shape: List[int]

class ImageEmbeddingResponse(BaseModel):
    embeddings: List[List[float]]
    shape: List[int]

async def load_model():
    """Load the SigLIP model and processor"""
    global model, processor, device, model_loaded
    
    try:
        logger.info("Starting model loading process...")
        
        # Determine device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {device}")
        
        # Set model cache directory
        cache_dir = os.environ.get('TRANSFORMERS_CACHE', '/app/model_cache')
        logger.info(f"Using model cache directory: {cache_dir}")
        
        # Load model and processor with retry logic
        model_name = "google/siglip-so400m-patch14-384"
        max_retries = 3
        
        for attempt in range(max_retries):
            try:
                logger.info(f"Loading model attempt {attempt + 1}/{max_retries}")
                
                processor = AutoProcessor.from_pretrained(
                    model_name,
                    cache_dir=cache_dir,
                    local_files_only=False
                )
                
                model = AutoModel.from_pretrained(
                    model_name,
                    cache_dir=cache_dir,
                    local_files_only=False,
                    torch_dtype=torch.float16 if device.type == "cuda" else torch.float32
                )
                
                model.to(device)
                model.eval()
                
                logger.info("Model loaded successfully!")
                model_loaded = True
                break
                
            except Exception as e:
                logger.error(f"Attempt {attempt + 1} failed: {str(e)}")
                if attempt < max_retries - 1:
                    logger.info(f"Retrying in 5 seconds...")
                    await asyncio.sleep(5)
                else:
                    raise e
        
        # Test the model with a dummy input
        logger.info("Testing model with dummy input...")
        test_text = "test"
        inputs = processor(text=[test_text], return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            test_features = model.get_text_features(**inputs)
            logger.info(f"Model test successful. Output shape: {test_features.shape}")
        
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        model_loaded = False
        # Don't raise here to allow the app to start, but mark model as not loaded

@app.on_event("startup")
async def startup_event():
    """Initialize the application on startup"""
    logger.info("Starting SigLIP Embedding API...")
    await load_model()

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "message": "SigLIP Embedding API is running",
        "model": "google/siglip-so400m-patch14-384",
        "device": str(device) if device else "not loaded",
        "model_loaded": model_loaded
    }

@app.get("/health")
async def health_check():
    """Detailed health check"""
    return {
        "status": "healthy" if model_loaded else "degraded",
        "model_loaded": model_loaded,
        "processor_loaded": processor is not None,
        "device": str(device) if device else "not loaded",
        "cuda_available": torch.cuda.is_available(),
        "gpu_memory": torch.cuda.get_device_properties(0).total_memory if torch.cuda.is_available() else None
    }

@app.post("/embed/text", response_model=TextEmbeddingResponse)
async def embed_text(request: TextRequest):
    """Generate embeddings for text input"""
    if not model_loaded or model is None or processor is None:
        raise HTTPException(
            status_code=503, 
            detail="Model not loaded. Please check /health endpoint for details."
        )
    
    try:
        # Handle both single string and list of strings
        texts = request.text if isinstance(request.text, list) else [request.text]
        
        # Validate input
        if not texts or any(not isinstance(t, str) or not t.strip() for t in texts):
            raise HTTPException(status_code=400, detail="Invalid text input")
        
        # Process texts
        inputs = processor(text=texts, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Generate embeddings
        with torch.no_grad():
            text_features = model.get_text_features(**inputs)
            # Normalize embeddings
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        # Convert to list
        embeddings = text_features.cpu().numpy().tolist()
        
        return TextEmbeddingResponse(
            embeddings=embeddings,
            shape=list(text_features.shape)
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating text embeddings: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error generating embeddings: {str(e)}")

@app.post("/embed/image", response_model=ImageEmbeddingResponse)
async def embed_image(file: UploadFile = File(...)):
    """Generate embeddings for image input"""
    if not model_loaded or model is None or processor is None:
        raise HTTPException(
            status_code=503, 
            detail="Model not loaded. Please check /health endpoint for details."
        )
    
    try:
        # Validate file type
        if not file.content_type or not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Read and process image
        image_data = await file.read()
        if len(image_data) == 0:
            raise HTTPException(status_code=400, detail="Empty image file")
            
        image = Image.open(io.BytesIO(image_data)).convert("RGB")
        
        # Process image
        inputs = processor(images=image, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Generate embeddings
        with torch.no_grad():
            image_features = model.get_image_features(**inputs)
            # Normalize embeddings
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        
        # Convert to list
        embeddings = image_features.cpu().numpy().tolist()
        
        return ImageEmbeddingResponse(
            embeddings=embeddings,
            shape=list(image_features.shape)
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating image embeddings: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error generating embeddings: {str(e)}")

@app.post("/embed/image_base64", response_model=ImageEmbeddingResponse)
async def embed_image_base64(image_data: dict):
    """Generate embeddings for base64 encoded image"""
    if not model_loaded or model is None or processor is None:
        raise HTTPException(
            status_code=503, 
            detail="Model not loaded. Please check /health endpoint for details."
        )
    
    try:
        # Decode base64 image
        base64_str = image_data.get("image")
        if not base64_str:
            raise HTTPException(status_code=400, detail="No image data provided")
        
        # Remove data URL prefix if present
        if base64_str.startswith("data:image"):
            base64_str = base64_str.split(",")[1]
        
        try:
            image_bytes = base64.b64decode(base64_str)
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid base64 image data")
            
        if len(image_bytes) == 0:
            raise HTTPException(status_code=400, detail="Empty image data")
            
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        
        # Process image
        inputs = processor(images=image, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Generate embeddings
        with torch.no_grad():
            image_features = model.get_image_features(**inputs)
            # Normalize embeddings
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        
        # Convert to list
        embeddings = image_features.cpu().numpy().tolist()
        
        return ImageEmbeddingResponse(
            embeddings=embeddings,
            shape=list(image_features.shape)
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating image embeddings: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error generating embeddings: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8000,
        log_level="info",
        access_log=True
    )
