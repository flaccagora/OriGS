# Use NVIDIA CUDA base image matching the PyTorch CUDA version (11.8)
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    wget \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    vim \
    && rm -rf /var/lib/apt/lists/*

# Install Miniconda
ENV CONDA_DIR=/opt/conda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p $CONDA_DIR && \
    rm ~/miniconda.sh

# Add conda to PATH
ENV PATH=$CONDA_DIR/bin:$PATH

# Initialize conda
RUN conda init bash

# Copy the current directory contents into the container at /app
WORKDIR /app
COPY . /app

# Ensure conda Terms of Service are accepted
RUN conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
RUN conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

# Create the conda environment 'origs' as specified in install.sh
# We include the GCC compilers in the environment creation
RUN conda create -n origs gcc_linux-64=9 gxx_linux-64=9 python=3.10 numpy=1.26.4 -y

# Use conda run -n origs for subsequent commands to ensure environment is active
SHELL ["conda", "run", "-n", "origs", "/bin/bash", "-c"]

# Set compiler environment variables to use the conda-installed compilers
# These variables match what is set in install.sh: $CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gcc
ENV CC=/opt/conda/envs/origs/bin/x86_64-conda-linux-gnu-gcc
ENV CXX=/opt/conda/envs/origs/bin/x86_64-conda-linux-gnu-g++
ENV CPP=/opt/conda/envs/origs/bin/x86_64-conda-linux-gnu-g++

# Verify compilers
RUN $CC --version && $CXX --version

# --- Begin installation steps from install.sh ---

# Install PyTorch and related libs
RUN pip install numpy==1.26.4 && \
    # conda install pytorch==2.1.0 torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y && \
    pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118 \
    conda install fvcore iopath -c fvcore -c iopath -c conda-forge -y && \
    conda install nvidiacub -c bottler -y && \
    # conda install pytorch3d -c pytorch3d -y && \
    pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable" --no-build-isolation
    pip install pyg_lib torch_scatter torch_geometric torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.1.0+cu118.html

# Install xformers and requirements
RUN conda install xformers -c xformers -y
RUN pip install chumpy --no-build-isolation
RUN pip install -r requirements.txt
RUN pip install numpy==1.26.4

RUN pip uninstall torch torchvision torchaudio -y && \
    pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118
# Install Gaussian Splatting / Rendering packages
# These require the previously set CC/CXX variables to compile correctly
RUN pip install lib_render/simple-knn --no-build-isolation && \
    pip install lib_render/diff-gaussian-rasterization-alphadep-add3 --no-build-isolation && \
    pip install lib_render/diff-gaussian-rasterization-alphadep --no-build-isolation && \
    pip install lib_render/gof-diff-gaussian-rasterization --no-build-isolation

# # Final dependency updates
RUN pip install numpy==1.26.4 && \
    pip install -U scikit-learn && \
    pip install -U scipy && \
    pip install opencv-python==4.10.0.84 && \
    pip install mmcv-full==1.7.2

# --- End installation steps ---

RUN pip install -r jax_requirements.txt

# Set the default shell to activate the environment when entering the container
RUN echo "conda activate origs" >> ~/.bashrc
SHELL ["/bin/bash", "--login", "-c"]

# Default command
CMD ["/bin/bash"]
