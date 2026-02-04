# Minimal Pixel Meanflow

Train a simple pixel diffusion model that generates images in a single step.

This is an unofficial implementation of [One-step Latent-free Image Generation with Pixel Mean Flows](https://arxiv.org/pdf/2601.22158),
that uses pytorch.

Requires [uv](https://docs.astral.sh/uv/`)

# Train a model

`uv run main.py --conf.should_compile true --conf.dtype_str bfloat16 --conf.device_str cuda train`

This trains a mini ViT with a patch size of 8 on imagenet1k 64x64 for 1000 epochs.
This requires 16 GB of gpu memory and an nvidia gpu.
Intermediate generation outputs are saved in out/..../artifacts.

Here is what the generated samples look like after 54 epochs:

<img width="1054" height="1054" alt="272500" src="https://github.com/user-attachments/assets/2fbd0655-7c6d-43a5-953c-ea32aab157ac" />
