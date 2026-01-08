# Docker Setup for OriGS

This directory contains the Docker configuration to set up the `origs` environment, mirroring the steps in `install.sh`.

## Prerequisites

- [Docker](https://docs.docker.com/get-docker/) installed.
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) installed (required for GPU support).

## Building the Image

To build the Docker image, run the following command from the root of the repository:

```bash
docker build -t origs-env .
```

This process may take some time as it downloads dependencies and compiles CUDA extensions.

## Running the Container

To launch the container with GPU support and mount the current directory:

```bash
docker run --gpus all -it --rm \
    -v $(pwd):/app \
    origs-env
```

- `--gpus all`: Enables access to all GPUs.
- `-it`: Runs in interactive mode.
- `--rm`: Removes the container after exit.
- `-v $(pwd):/app`: Mounts your current directory to `/app` inside the container, allowing you to edit files on your host and run them inside the container.

## Persisting Changes

The command above uses `--rm`, which removes the container environment when you exit.

- **Code/Data Persistence**: Any changes you make to files inside `/app` **are saved** because that directory is mounted from your host machine (`-v $(pwd):/app`).
- **System/Environment Persistence**: Any changes to the system (e.g., `pip install new-package`, `apt-get install`) are **lost** when you exit if you use `--rm`.

To keep system-level changes, you have three options:

1.  **Update Dockerfile (Recommended)**: Add the new content to `Dockerfile` and rebuild. This ensures your environment is reproducible.
2.  **Remove `--rm`**: Run the container without the `--rm` flag. When you exit, the container stops but isn't deleted. You can resume it later with `docker start -ai <container_name>`.
3.  **Commit to New Image**: If you have made complex changes interactively, you can save the container's state as a new image:
    ```bash
    # In a new terminal (while container is running or after stopping without --rm)
    docker commit <container_id_or_name> origs-env-custom
    ```

## Environment

The environment `origs` is automatically activated when you enter the container. You should see `(origs)` in your command prompt.

You can verify the installation by running:
```bash
python -c "import torch; print(torch.cuda.is_available())"
```

## Troubleshooting

- **Out of Memory during Build**: Docker default memory limits might be too low for compiling some extensions. You can try increasing docker memory or ensuring you have enough swap.
- **Permission Errors**: If you encounter file permission issues with the mounted volume, make sure the user IDs match or work inside the container as root (default).
