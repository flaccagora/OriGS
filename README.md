# [NeurIPS 2025]: Orientation-anchored Hyper-Gaussian for 4D Reconstruction from Casual Videos

[![arXiv](https://img.shields.io/badge/arXiv-2509.23492-b31b1b.svg)](https://arxiv.org/abs/2509.23492)

## Install

The installation process is identical to [MoSca](https://github.com/JiahuiLei/MoSca):

1. Simply run the following command. This script assumes you have an Ubuntu environment and Anaconda installed. The CUDA version used is 11.8. You may need to tweak the script to fit your environment.
    ```bash
    bash install.sh
    ```

2. Download from [here](https://drive.google.com/file/d/15tveiv7ZkvBBAN3qkkB7Zfky9d7vSqLD/view?usp=sharing) the checkpoints for 2D foundational models if they are not Hugging Face downloadable.

    **WARNING**: By downloading these checkpoints, you must agree and obey the original license from the original authors ([RAFT](https://github.com/princeton-vl/RAFT), [SpaTracker](https://github.com/henry123-boy/SpaTracker), and [TAPNet](https://github.com/google-deepmind/tapnet)). Unzip the weights into the following file structure:
    ```bash
    ProjRoot/weights
    ├── raft_models
    │   ├── raft-things.pth
    │   └── ...
    ├── spaT_final.pth
    └── tapnet
        └── bootstapir_checkpoint_v2.pt
    ```

## Get Started

### Prepare Your Data

To reconstruct your own video, organize your data in the following structure:

```bash
demo/your_scene_name/
└── images/
    ├── 00000.jpg  # or .png
    ├── 00001.jpg
    ├── 00002.jpg
    └── ...
```

**Requirements**:
- **Image format**: `.jpg` or `.png`
- **Naming convention**: Sequential numbering with zero-padding (e.g., `00000.jpg`, `00001.jpg`, ...)
- **Frame rate**: 20-30 FPS recommended for dynamic scenes
- **Resolution**: The code will automatically resize images if needed
- **Video extraction**: If you have a video file, extract frames using:
  ```bash
  ffmpeg -i your_video.mp4 -qscale:v 2 demo/your_scene_name/images/%05d.jpg
  ```

### Quick Start with Demo

We provide demo scenes in the `demo/` directory for quick testing. Each scene follows the structure above with images under `demo/scene_name/images/`.

#### Step 1: Precompute 2D Priors

```bash
python origs_precompute.py \
    --cfg ./profile/demo/demo_prep.yaml \
    --ws ./demo/lucia
```

This step runs off-the-shelf 2D foundational models (depth estimation, optical flow, tracking) on the input images.

#### Step 2: 4D Reconstruction

```bash
python origs_reconstruct.py \
    --cfg ./profile/demo/demo_fit.yaml \
    --ws ./demo/lucia
```

This step performs the OriGS reconstruction pipeline.

#### All-in-one Script

You can also run both steps using the provided script:

```bash
bash demo.sh
```


## Configuration

- `demo_prep.yaml`: Controls 2D prior computation (depth model, tracking model, optical flow settings)
- `demo_fit.yaml`: Controls reconstruction pipeline (bundle adjustment parameters, scaffold settings, photometric optimization)


## Acknowledgement

This work builds upon [MoSca](https://github.com/JiahuiLei/MoSca). If you use this code, please cite both OriGS and MoSca:

```tex
@inproceedings{wu2025orientation,
  title={Orientation-anchored Hyper-Gaussian for 4D Reconstruction from Casual Videos},
  author={Wu, Junyi and Tao, Jiachen and Wang, Haoxuan and Liu, Gaowen and Kompella, Ramana Rao and Yan, Yan},
  booktitle={NeurIPS},
  year={2025}
}

@inproceedings{lei2025mosca,
  title={Mosca: Dynamic gaussian fusion from casual videos via 4d motion scaffolds},
  author={Lei, Jiahui and Weng, Yijia and Harley, Adam W and Guibas, Leonidas and Daniilidis, Kostas},
  booktitle={CVPR},
  year={2025}
}
```
