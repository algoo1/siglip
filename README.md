# SigLIP Embedding API

A FastAPI server that provides text and image embedding endpoints using Google's SigLIP (Sigmoid Loss for Language Image Pre-training) model `google/siglip-so400m-patch14-384`.

## Features

- **Text Embeddings**: Generate embeddings for single text strings or batches of text
- **Image Embeddings**: Generate embeddings for images via file upload or base64 encoding
- **GPU Support**: Automatically detects and uses CUDA if available
- **Normalized Embeddings**: All embeddings are L2 normalized for similarity calculations
- **Health Checks**: Built-in health monitoring endpoints
- **CORS Enabled**: Ready for web applications

## API Endpoints

### Health Check
- `GET /` - Basic health check
- `GET /health` - Detailed health status

### Embeddings
- `POST /embed/text` - Generate text embeddings
- `POST /embed/image` - Generate image embeddings (file upload)
- `POST /embed/image_base64` - Generate image embeddings (base64)

## Local Development

### Prerequisites
- Python 3.8+
- CUDA-compatible GPU (optional, but recommended)

### Installation

1. Clone the repository:
```bash
git clone <your-github-repo-url>
cd siglip
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the server:
```bash
python main.py
```

The server will start on `http://localhost:8000`

### API Usage Examples

#### Text Embeddings
```bash
curl -X POST "http://localhost:8000/embed/text" \
     -H "Content-Type: application/json" \
     -d '{"text": "Hello world"}'
```

#### Image Embeddings (File Upload)
```bash
curl -X POST "http://localhost:8000/embed/image" \
     -F "file=@your_image.jpg"
```

#### Image Embeddings (Base64)
```bash
curl -X POST "http://localhost:8000/embed/image_base64" \
     -H "Content-Type: application/json" \
     -d '{"image": "<base64_encoded_image>"}'
```

## RunPod Deployment

### Step 1: Push to GitHub

1. Create a new GitHub repository
2. Push your code:
```bash
git init
git add .
git commit -m "Initial commit: SigLIP embedding API"
git branch -M main
git remote add origin <your-github-repo-url>
git push -u origin main
```

### Step 2: Deploy on RunPod

1. **Create RunPod Account**
   - Sign up at [runpod.io](https://runpod.io)
   - Add credits to your account

2. **Create a New Pod**
   - Go to "Pods" in the RunPod dashboard
   - Click "Deploy"
   - Select a GPU instance (recommended: RTX 3090 or better)

3. **Configure Container**
   - **Container Image**: Use a custom Docker image or:
   - **Template**: Select "Custom"
   - **Docker Image**: `nvidia/cuda:11.8-runtime-ubuntu20.04`
   - **Container Disk**: At least 20GB
   - **Volume**: Optional, for persistent storage

4. **Environment Setup**
   Add these environment variables:
   ```
   GITHUB_REPO=<your-github-repo-url>
   ```

5. **Startup Script**
   Add this startup script:
   ```bash
   #!/bin/bash
   apt-get update && apt-get install -y git python3 python3-pip curl
   git clone $GITHUB_REPO /app
   cd /app
   pip3 install -r requirements.txt
   python3 main.py
   ```

6. **Ports**
   - Expose port `8000`
   - Set as HTTP port

7. **Deploy**
   - Click "Deploy"
   - Wait for the pod to start (may take 5-10 minutes for first-time model download)

### Step 3: Access Your API

Once deployed, you'll get a public URL like:
`https://your-pod-id-8000.proxy.runpod.net`

Test the deployment:
```bash
curl https://your-pod-id-8000.proxy.runpod.net/health
```

## Alternative RunPod Deployment (Docker)

For more control, you can build and push a Docker image:

1. **Build Docker Image**:
```bash
docker build -t your-username/siglip-api .
```

2. **Push to Docker Hub**:
```bash
docker push your-username/siglip-api
```

3. **Deploy on RunPod**:
   - Use your Docker image: `your-username/siglip-api`
   - Expose port 8000
   - No additional setup required

## Model Information

- **Model**: `google/siglip-so400m-patch14-384`
- **Input Resolution**: 384x384 pixels for images
- **Embedding Dimension**: 1152
- **Text Max Length**: 77 tokens
- **Normalization**: L2 normalized embeddings

## Performance Notes

- **GPU Memory**: ~4GB VRAM required
- **Model Loading**: ~30-60 seconds on first startup
- **Inference Speed**: 
  - Text: ~10-50ms per request
  - Image: ~50-200ms per request

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce batch size or use CPU inference
   - Ensure sufficient GPU memory (4GB+)

2. **Model Download Timeout**
   - Increase timeout settings
   - Check internet connection

3. **Port Access Issues**
   - Ensure port 8000 is properly exposed
   - Check firewall settings

### Logs

Check application logs for debugging:
```bash
# In RunPod terminal
tail -f /var/log/your-app.log
```

## License

This project is open source. Please check the model license for `google/siglip-so400m-patch14-384`.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## Support

For issues and questions:
- Create an issue on GitHub
- Check RunPod documentation for deployment issues