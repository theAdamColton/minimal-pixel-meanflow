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

# Implementation Details

### Transformer

I use DerfNorm, Box-RoPE-2d which is similar to dinov3, and QK normalization. Note that ALL Normalization layers
were replaced with DerfNorm.

### Perceptual losses

They compute perceptual losses on predicted samples from timesteps less than some threshold t_thr.
I instead compute perceptual losses on a fixed proportion of lowest timesteps.
This allows me to torch.compile the entire loss function.

My Convnextv2 loss seems slightly broken right now. The LPIPs seems to work fine.

Before computing Convnextv2 or LPIPs I random resized crop each individual image.

I add an additional REPA loss in the style of iREPA (https://arxiv.org/pdf/2512.10794). Hidden states
come from an intermediate layer from the forward pass used to compute the instantaneous velocity prediction.

### Classifier-Free-Guidance

The diffusion model accepts a CFG scale as an input. During training it estimates the guided velocity and mean trajectory. Unlike iMF and pMF I do not use CFG interval as a model input and instead only apply CFG to samples that are noised within some CFG interval (by default 0.1 to 0.7]).

### Floating Point Precision and JVP

I couldn't seem to figure out how to use torch's autocast when using JVP.
So I use an extra forward pass to predict the mean trajectory u.
