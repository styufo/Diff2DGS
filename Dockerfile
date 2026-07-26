FROM pytorch/pytorch:2.1.2-cuda11.8-cudnn8-devel

ENV DEBIAN_FRONTEND=noninteractive \
    CUDA_HOME=/usr/local/cuda \
    FORCE_CUDA=1 \
    TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6;8.9"

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        curl \
        ffmpeg \
        git \
        libgl1 \
        libglib2.0-0 \
        ninja-build \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace/Diff2DGS
COPY . .

RUN python -m pip install --no-cache-dir --upgrade pip "setuptools<70" wheel \
    && python -m pip install --no-cache-dir torchvision==0.16.2+cu118 \
        --index-url https://download.pytorch.org/whl/cu118 \
    && python -m pip install --no-cache-dir -r requirements.txt \
    && python -m pip install --no-cache-dir -e . \
    && python -m pip install --no-cache-dir --no-build-isolation ./reconstruction/submodules/simple-knn \
    && python -m pip install --no-cache-dir --no-build-isolation ./reconstruction/submodules/diff-surfel-rasterization

ENTRYPOINT ["diff2dgs"]
