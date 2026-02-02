# Minimal Pixel Meanflow

Train a simple pixel diffusion model that generates samples in a single step.

Requires `uv`

# Train a model

`uv run main.py --conf.should_compile true --conf.num_workers 8 --conf.dtype_str bfloat16 --conf.device_str cuda train`

This trains a mini ViT with a patch size of 4 on imagenet1k 64x64 for 1000 epochs.
This requires 16 GB of gpu memory and an nvidia gpu.
Intermediate generation outputs are saved in out/..../artifacts.
