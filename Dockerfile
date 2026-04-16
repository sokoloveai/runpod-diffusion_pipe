# Use CUDA base image
FROM nvidia/cuda:13.0.0-cudnn-devel-ubuntu24.04 AS base

# Consolidated environment variables
ENV DEBIAN_FRONTEND=noninteractive \
   PIP_PREFER_BINARY=1 \
   PIP_BREAK_SYSTEM_PACKAGES=1 \
   PYTHONUNBUFFERED=1 \
   CMAKE_BUILD_PARALLEL_LEVEL=8

# Install dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
   python3 python3-pip python3-venv curl zip git git-lfs wget vim libgl1 libglib2.0-0 \
   python3-dev build-essential gcc \
   && ln -sf /usr/bin/python3 /usr/bin/python \
   && ln -sf /usr/bin/pip3 /usr/bin/pip \
   && apt-get clean \
   && rm -rf /var/lib/apt/lists/*

# Install Python packages
RUN pip install --no-cache-dir gdown jupyterlab jupyterlab-lsp \
    jupyter-server jupyter-server-terminals \
    ipykernel jupyterlab_code_formatter huggingface_hub[cli] \
    ninja packaging

# Create the final image
FROM base AS final

# PyTorch (latest for cu130)
RUN pip install --no-cache-dir \
    torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu130

# Flash-attn from prebuilt wheel (cu130 + torch2.11 + Python 3.12)
RUN pip install --no-cache-dir \
    https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.9.4/flash_attn-2.7.4%2Bcu130torch2.11-cp312-cp312-manylinux_2_24_x86_64.manylinux_2_28_x86_64.whl

# Clone diffusion-pipe and install ALL requirements (flash-attn already installed above)
RUN git clone --recurse-submodules https://github.com/tdrussell/diffusion-pipe /diffusion_pipe
RUN pip install --no-cache-dir -r /diffusion_pipe/requirements.txt

COPY src/start_script.sh /start_script.sh
RUN chmod +x /start_script.sh
CMD ["/start_script.sh"]
