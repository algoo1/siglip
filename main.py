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

# Configure logging
logging.basicConfig(level=logging.INFO)
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

class TextRequest(BaseModel):
    text: Union[str, List[str]]

class TextEmbeddingResponse(BaseModel):
    embeddings: List[List[float]]
    shape: List[int]

class ImageEmbeddingResponse(BaseModel):
    embeddings: List[List[float]]
    shape: List[int]

@app.on_event("startup")
async def load_model():
    """Load the SigLIP model and processor on startup"""
    global model, processor, device
    
    try:
        logger.info("Loading SigLIP model...")
        
        # Determine device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {device}")
        
        # Load model and processor
        model_name = "google/siglip-so400m-patch14-384"
        processor = AutoProcessor.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        model.to(device)
        model.eval()
        
        logger.info("Model loaded successfully!")
        
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise e

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "message": "SigLIP Embedding API is running",
        "model": "google/siglip-so400m-patch14-384",
        "device": str(device) if device else "not loaded"
    }

@app.get("/health")
async def health_check():
    """Detailed health check"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "processor_loaded": processor is not None,
        "device": str(device) if device else "not loaded"
    }

@app.post("/embed/text", response_model=TextEmbeddingResponse)
async def embed_text(request: TextRequest):
    """Generate embeddings for text input"""
    if model is None or processor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Handle both single string and list of strings
        texts = request.text if isinstance(request.text, list) else [request.text]
        
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
        
    except Exception as e:
        logger.error(f"Error generating text embeddings: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error generating embeddings: {str(e)}")

@app.post("/embed/image", response_model=ImageEmbeddingResponse)
async def embed_image(file: UploadFile = File(...)):
    """Generate embeddings for image input"""
    if model is None or processor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Read and process image
        image_data = await file.read()
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
        
    except Exception as e:
        logger.error(f"Error generating image embeddings: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error generating embeddings: {str(e)}")

@app.post("/embed/image_base64", response_model=ImageEmbeddingResponse)
async def embed_image_base64(image_data: dict):
    """Generate embeddings for base64 encoded image"""
    if model is None or processor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Decode base64 image
        base64_str = image_data.get("image")
        if not base64_str:
            raise HTTPException(status_code=400, detail="No image data provided")
        
        # Remove data URL prefix if present
        if base64_str.startswith("data:image"):
            base64_str = base64_str.split(",")[1]
        
        image_bytes = base64.b64decode(base64_str)
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
        
    except Exception as e:
        logger.error(f"Error generating image embeddings: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error generating embeddings: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)