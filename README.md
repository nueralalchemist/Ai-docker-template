# 🚀 Ubuntu AI Developer Docker Container

A powerful, GPU-enabled Docker environment based on Ubuntu + CUDA for AI/ML developers.  
This container includes **PyTorch, TensorFlow, Hugging Face Transformers, FastAPI, JupyterLab**, and more — all preconfigured and ready to use.

---

## 📦 What's Inside?

| Category             | Tools & Frameworks                                                                 |
|----------------------|-------------------------------------------------------------------------------------|
| 🧠 Deep Learning      | PyTorch (CUDA 12.1), TensorFlow (GPU), TorchAudio, TorchVision                     |
| 🔤 NLP & Transformers | HuggingFace Transformers, Datasets, SentenceTransformers, spaCy, NLTK              |
| 📊 Data & ML          | scikit-learn, pandas, numpy, matplotlib, seaborn                                   |
| 🎛️ Utilities          | OpenCV, tqdm, ffmpeg, ONNX, ONNXRuntime-GPU                                        |
| 🧪 Serving/Dev Tools  | Flask, FastAPI, Uvicorn, Gradio, JupyterLab, Jupyter Notebook                      |
| ⚙️ System Tools       | nano, git, curl, wget, unzip, htop, build-essential, python3.10, pip, venv         |

---

## 🛠️ Prerequisites

- Docker installed ([Install Docker](https://docs.docker.com/get-docker/))
- **NVIDIA GPU & NVIDIA Container Toolkit** installed ([Guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html))

---

## 🧱 Build the Image

Clone this repository:

```bash
git clone https://github.com/nueralalchemist/ubuntu-Ai-docker-template.git
cd Ai-docker-template
```

Then build the Docker image:

```bash
docker build -t Ai-docker .
```

---

## 🚀 Run the Container

Use the following command to start the container with GPU support and port bindings:

```bash
docker run -it --rm \
  --gpus all \
  -p 8888:8888 \        # JupyterLab
  -p 7860:7860 \        # Gradio
  -p 8000:8000 \        # FastAPI / Flask
  -v $(pwd):/home/aiuser/app \
  ubuntu-ai-dev
```

💡 The app directory will be mounted at `/home/aiuser/app`.

---

## 📂 Folder Structure

```bash
Ai-docker-template/
│
├── Dockerfile            # Docker build file
├── README.md             # Project documentation
```

---

## 🌐 Accessing Services

| Service      | URL                                  | Description                       |
|--------------|--------------------------------------|-----------------------------------|
| JupyterLab   | http://localhost:8888                | Interactive notebooks             |
| Gradio App   | http://localhost:7860                | ML model demos & interfaces       |
| FastAPI App  | http://localhost:8000/docs           | Auto-generated API docs (Swagger) |
| Flask App    | http://localhost:8000                | Flask-based web services          |

---

## 🧪 Sample Use

After starting the container:

1. Open JupyterLab in your browser at `http://localhost:8888`
2. Create new notebooks or Python files in `/home/aiuser/app`
3. Run ML/DL code with GPU acceleration

---

## 🔐 Notes

- Default container user: `aiuser`
- Home directory: `/home/aiuser`
- Use volume mounts to persist data/code
- Container has no password, and it runs as non-root for security

---

## 🧰 Extras

You can extend this Dockerfile with:
- VS Code Remote Containers (`devcontainer.json`)
- Docker Compose for multi-container workflows
- Preloading HuggingFace models or datasets

---

### 🚀 Sample `docker run` with GPU access:

```bash
docker build -t ubuntu-ai-dev .

docker run -it --rm \
  --gpus all \
  -p 8888:8888 -p 7860:7860 -p 8000:8000 \
  -v $(pwd):/home/aiuser/app \
  ubuntu-ai-dev
```

> 🧠 Make sure your host has **NVIDIA Container Toolkit** installed to use `--gpus all`  
> Install using:
```bash
sudo apt install nvidia-container-toolkit
sudo systemctl restart docker
```

---

### 🧰 Key Features:

- CUDA 12.2 + cuDNN 8 runtime
- Python 3.10 + pip
- GPU-enabled PyTorch, TensorFlow
- NLP tools: HuggingFace, spaCy, NLTK
- Visualization: Matplotlib, Seaborn
- Model serving support: Flask, FastAPI, Gradio
- JupyterLab environment

---
