GPU_ID=1

# CUDA_VISIBLE_DEVICES=$GPU_ID python mosca_precompute.py --cfg ./profile/demo/demo_prep.yaml --ws ./demo/duck
# CUDA_VISIBLE_DEVICES=$GPU_ID python mosca_reconstruct.py --cfg ./profile/demo/demo_fit.yaml --ws ./demo/duck

# CUDA_VISIBLE_DEVICES=$GPU_ID python mosca_precompute.py --cfg ./profile/demo/demo_prep.yaml --ws ./demo/shiba --tap_mode=bootstapir --boundary_enhance_th=-1.0
# CUDA_VISIBLE_DEVICES=$GPU_ID python mosca_reconstruct.py --cfg ./profile/demo/demo_fit.yaml --ws ./demo/shiba

# CUDA_VISIBLE_DEVICES=$GPU_ID python mosca_precompute.py --cfg ./profile/demo/demo_prep.yaml --ws ./demo/breakdance-flare --dep_mode=uni --tap_mode=bootstapir --boundary_enhance_th=-1.0
# CUDA_VISIBLE_DEVICES=$GPU_ID python mosca_reconstruct.py --cfg ./profile/demo/demo_fit.yaml --ws ./demo/breakdance-flare

# CUDA_VISIBLE_DEVICES=$GPU_ID python mosca_precompute.py --cfg ./profile/demo/demo_prep.yaml --ws ./demo/train --dep_mode=uni --tap_mode=bootstapir --boundary_enhance_th=-1.0
# CUDA_VISIBLE_DEVICES=$GPU_ID python mosca_reconstruct.py --cfg ./profile/demo/demo_fit.yaml --ws ./demo/train

# CUDA_VISIBLE_DEVICES=$GPU_ID python mosca_precompute.py --cfg ./profile/demo/demo_prep.yaml --ws ./demo/elephant
# CUDA_VISIBLE_DEVICES=$GPU_ID python mosca_reconstruct.py --cfg ./profile/demo/demo_fit.yaml --ws ./demo/elephant

CUDA_VISIBLE_DEVICES=$GPU_ID python mosca_reconstruct.py --cfg ./profile/demo/demo_fit.yaml --ws ./demo/drift-turn

# CUDA_VISIBLE_DEVICES=$GPU_ID python mosca_reconstruct.py --cfg ./profile/demo/demo_fit.yaml --ws ./demo/swing

