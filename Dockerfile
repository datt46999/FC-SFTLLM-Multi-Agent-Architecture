FROM python:3.10-slim

WORKDIR /src

ENV DEBIAN_FRONTEND=noninteractive

# System dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --upgrade pip setuptools wheel

# Install PyTorch trước với CUDA
RUN pip install torch==2.1.0 --index-url https://download.pytorch.org/whl/cu121

# Install các thư viện fine-tuning
RUN pip install \
    transformers \
    peft \
    trl \
    datasets \
    accelerate \
    bitsandbytes \
    scipy \
    sentencepiece

# Copy source code
COPY . .

CMD ["python3", "SFTtrainer.py"]