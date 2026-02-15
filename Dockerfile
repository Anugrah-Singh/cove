# Use NVIDIA CUDA base for GPU acceleration
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# Set non-interactive mode for apt-get
ENV DEBIAN_FRONTEND=noninteractive

# Update and install Python + System Deps (OpenCV requires libgl1)
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Upgrade pip
RUN python3 -m pip install --upgrade pip

# Install Python dependencies
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt
# Override with GPU-accelerated versions for Docker
RUN pip3 install --no-cache-dir onnxruntime-gpu faiss-gpu

# Copy the entire project
COPY . .

# Expose API port
EXPOSE 8000

# Default command: Run the API server
CMD ["python3", "server.py"]
