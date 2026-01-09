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



## Building without Sudo (Root)

If you are on a cluster or machine where you do not have root access (sudo), you have two main options:

### Option 1: Fakeroot
Modern versions of Singularity/Apptainer support the `--fakeroot` flag, which simulates root privileges. This requires the system administrator to have configured `/etc/subuid` and `/etc/subgid` mappings for your user.

```bash
singularity build --fakeroot origs.sif Singularity.def
```
or with Apptainer:

```bash
apptainer build --fakeroot --sandbox --nv --writable origs.sif Singularity.def
```

### Option 2: Build from Docker Image
If you have access to Docker on your local machine (or can build the Docker image elsewhere), you can build the Docker image locally and then convert it to Singularity.

1.  **Build Docker Image**:
    ```bash
    docker build -t origs-env .
    ```

2.  **Convert to Singularity**:
    *   **From local Daemon**:
        ```bash
        singularity build origs.sif docker-daemon://origs-env:latest
        ```
    *   **From an Archive**:
        If you need to transfer it to a cluster:
        ```bash
        # On local machine
        docker save origs-env -o origs-env.tar

        # Transfer origs-env.tar to cluster...

        # On cluster
        singularity build origs.sif docker-archive://origs-env.tar
        ```

This process will:
1.  Pull the base Docker image (NVIDIA CUDA 11.8).
2.  Copy local `requirements.txt` and `lib_render/` into the image.
3.  Install all dependencies and compile the custom CUDA extensions.

## Running the Container

To use the container with GPU support (`--nv`) and bind your current directory (`-B` is often implicitly handled, but explicit binding ensures access):

```bash
singularity shell --nv --fakeroot --writable --home $HOME  origs.sif 
source /opt/conda/etc/profile.d/conda.sh
conda activate origs
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
