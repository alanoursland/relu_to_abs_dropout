# Docker Setup & Container Creation Guide

This guide explains how to install Docker and build a container image for running `container_main.py`.

---

## 1. Install Docker

Docker provides a consistent environment for running your experiments across different machines.

### **Windows**

* **Requires WSL 2**

  1. Install [WSL 2](https://learn.microsoft.com/en-us/windows/wsl/install).
  2. Download [Docker Desktop for Windows](https://www.docker.com/products/docker-desktop/).
  3. During setup, enable the **WSL 2 backend**.
  4. Verify installation:

     ```powershell
     docker --version
     ```

### **macOS**

* Download [Docker Desktop for Mac](https://www.docker.com/products/docker-desktop/).
* Follow installation prompts (drag to Applications).
* Verify:

  ```bash
  docker --version
  ```

### **Linux (Ubuntu/Debian example)**

```bash
sudo apt update
sudo apt install -y ca-certificates curl gnupg lsb-release

# Add Docker's official GPG key
sudo mkdir -p /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg \
    | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg

# Add repository
echo \
  "deb [arch=$(dpkg --print-architecture) \
  signed-by=/etc/apt/keyrings/docker.gpg] \
  https://download.docker.com/linux/ubuntu \
  $(lsb_release -cs) stable" \
  | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

# Install Docker Engine
sudo apt update
sudo apt install -y docker-ce docker-ce-cli containerd.io

# Verify
docker --version
```

---

## 2. Prepare Your Project for Docker

Ensure your repo contains:

* `container_main.py`
* `requirements.txt` with all Python dependencies
* Your experiment code (`train.py`, `config.py`, `experiments.py`, etc.)
* **Optional:** `Dockerfile` (see below)

---

## 3. Create a `Dockerfile`

Example minimal `Dockerfile` for PyTorch + your project:

```dockerfile
# Start from a CUDA + PyTorch image (change version as needed)
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

# Set working directory
WORKDIR /app

# Copy dependency list and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project code
COPY . .

# Default command (can be overridden at runtime)
CMD ["python", "container_main.py", "--help"]
```

---

## 4. Build Your Container Image

From your project root:

```bash
docker build -t myproject:latest .
```

Verify:

```bash
docker images
```

---

## 5. Run Your Container Locally

Example (with local results folder mount):

```bash
docker run --rm \
    -e RESULT_URL="https://example.com/presigned_upload_url" \
    -v $(pwd)/results:/app/results \
    myproject:latest \
    --experiment resnet_cifar10 \
    --run_index 0 \
    --archive zip \
    --allow-missing-upload
```

**Notes:**

* `-e` sets environment variables.
* `-v` mounts local folders into the container (so results are saved locally).
* You can override the `CMD` in the Dockerfile by passing arguments after the image name.

---

## 6. Push to a Remote Registry (Optional)

If running on Salad or another cloud batch system, push your image:

```bash
# Tag for registry
docker tag myproject:latest my-dockerhub-username/myproject:latest

# Login to Docker Hub (or other registry)
docker login

# Push
docker push my-dockerhub-username/myproject:latest
```

---

## 7. Next Steps

* Integrate with your batch scheduler (e.g., Salad jobs, Kubernetes Jobs).
* Provide `RESULT_URL` per job for cloud uploads.
* Use `--run_index` to coordinate multi-run experiments.

---

If you want, I can also prepare a **GPU-enabled Dockerfile** that matches Salad’s recommended base images and supports CUDA 11.8 so your ResNet training runs without additional driver headaches. That would make your deployment smoother. Would you like me to do that?
