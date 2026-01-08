# Singularity Setup for OriGS

This directory contains the Singularity definition to set up the `origs` environment, mirroring the steps in `install.sh`.

## Prerequisites

- [Apptainer/Singularity](https://apptainer.org/docs/user/main/quick_start.html) installed.
- NVIDIA Driver and working GPU (for runtime).

## Building the Container (SIF)

To build the Singularity Image File (`.sif`), run the following command. Note that you need `sudo` to build from a recipe file.

```bash
sudo singularity build origs.sif Singularity.def
```

This process will:
1.  Pull the base Docker image (NVIDIA CUDA 11.8).
2.  Copy local `requirements.txt` and `lib_render/` into the image.
3.  Install all dependencies and compile the custom CUDA extensions.

## Running the Container

To use the container with GPU support (`--nv`) and bind your current directory (`-B` is often implicitly handled, but explicit binding ensures access):

```bash
singularity shell --nv origs.sif
```

Or to run a specific command:

```bash
singularity exec --nv origs.sif python origs_reconstruct.py ...
```

## Environment

The `origs` conda environment is effectively "activated" by default because the environment's `bin` directory is prepended to the `$PATH` in the `%environment` section.

You can verify the setup inside the container:
```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```
