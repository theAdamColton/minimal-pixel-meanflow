# Minimal Pixel Meanflow

Train a simple pixel diffusion model that generates images in a single step.

This is an unofficial implementation of [One-step Latent-free Image Generation with Pixel Mean Flows](https://arxiv.org/pdf/2601.22158),
that uses pytorch.

Requires [uv](https://docs.astral.sh/uv/`)

You can find the official implementation [Here](https://github.com/Lyy-iiis/pMF)

# Train a model

This command trains a ViT-S with a patch size of 32 on imagenet1k 256x256 for 500 epochs and requires 8 GB of device memory.

`uv run main.py --config conf/in1k-256-ViT-S-p32.yaml --conf.should_compile true train`

Intermediate generations are saved to `out/.../artifacts/`

# Implementation Details

### Complexity

During each training step the model is forwarded four times without gradients and one time with
gradients.

A forward passes with 2 x batch size computes the guided instantaneous velocity without gradients.
Another two forward passes without gradients to compute the meanflow JVP. The final forward
pass computes the predicted mean flow with gradients enabled.

During test time only one forward pass is necessary for guided generation.

### Transformer

I use DerfNorm, Box-RoPE-2d which is similar to dinov3, and QK normalization. Note that ALL Normalization layers are replaced with DerfNorm.

By default I pin all adaLN projections - the model reuses the same adaLN projection for all layers.

Unlike the official implementation, I do not use an auxiliary v-prediction head.

### Perceptual losses

They compute perceptual losses on predicted samples from timesteps less than some threshold t_thr.
I instead compute perceptual losses on a fixed proportion of lowest timesteps.
This allows me to torch.compile the entire loss function.

My Convnextv2 loss seems slightly broken right now. The LPIPs seems to work fine.

Before computing Convnextv2 or LPIPs I random resized crop each individual image.

I add an additional REPA loss in the style of iREPA (https://arxiv.org/pdf/2512.10794). Hidden states
come from an intermediate layer from the forward pass used to compute the instantaneous velocity prediction.

### REPA loss

A conv2d uses hidden states from the model's 6th layer to predict dinov3 states. pMF does not use REPA but I assume that REPA can speed up convergence for pMF models.

### Classifier-Free-Guidance

The diffusion model accepts a CFG scale as an input. Unlike traditional CFG, you do not need to
do two forward passes to guide sampling.

Unlike iMF and pMF I do not use the CFG interval as a model input.
Instead I only apply CFG to samples that are noised within some CFG interval (by default 0.1 to 0.7]). This reduces the number of forward passes by 1 compared to the iMF code.

### Floating Point Precision and JVP

I couldn't seem to figure out how to use torch's autocast when using JVP with gradients enabled.
So I compute JVP inside of a no_grad context and use an extra forward pass to predict the mean trajectory u.
