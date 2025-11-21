ulimit -n 8192
GPU_ID=5

CUDA_VISIBLE_DEVICES=$GPU_ID python origs_precompute.py --cfg ./profile/demo/demo_prep.yaml --ws ./demo/lucia
CUDA_VISIBLE_DEVICES=$GPU_ID python origs_reconstruct.py --cfg ./profile/demo/demo_fit.yaml --ws ./demo/lucia


