# Use NVIDIA's TensorRT base image
FROM nvcr.io/nvidia/tensorrt:23.10-py3

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu118

# Copy application code
COPY . .

# Environment variables
ENV PYTHONUNBUFFERED=1
ENV INFERENCE_BACKEND=pytorch

# Expose port
EXPOSE 8000

# Run command
CMD ["uvicorn", "api.server:app", "--host", "0.0.0.0", "--port", "8000"]
