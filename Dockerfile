# Base Image: CUDA-enabled Ubuntu for AI development
FROM nvidia/cuda:12.2.0-cudnn8-runtime-ubuntu22.04

# Maintainer
LABEL maintainer="Nueralalchemist"

# Environment Setup
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Kolkata
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Update & Install essentials
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    software-properties-common \
    wget \
    curl \
    git \
    ca-certificates \
    python3.10 \
    python3-pip \
    python3-venv \
    python3-dev \
    nano \
    unzip \
    htop \
    ffmpeg \
    libsm6 \
    libxext6 \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN python3 -m pip install --upgrade pip

# Install AI/ML Libraries
RUN pip install --no-cache-dir \
    torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 \
    tensorflow \
    transformers \
    datasets \
    accelerate \
    scikit-learn \
    pandas \
    numpy \
    matplotlib \
    seaborn \
    jupyterlab \
    notebook \
    ipywidgets \
    opencv-python \
    onnx onnxruntime-gpu \
    sentence-transformers \
    nltk \
    spacy \
    flask \
    fastapi \
    uvicorn \
    gradio \
    tqdm

# Optional: Download NLTK and spaCy models
RUN python3 -m nltk.downloader punkt \
 && python3 -m spacy download en_core_web_sm

# Create AI developer user
RUN useradd -ms /bin/bash aiuser
USER aiuser
WORKDIR /home/aiuser/app

# Copy local code if needed
# COPY ./your-code ./app

# Expose Jupyter, FastAPI/Flask, Gradio
EXPOSE 8888 7860 8000

# Default: launch JupyterLab
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]
