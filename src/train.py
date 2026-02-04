import json
import math
from dataclasses import asdict
import gc
from functools import partial
import functools
from pathlib import Path
from typing import Any, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torchvision
import wandb
from einops import rearrange, repeat
from torch.utils.data import DataLoader
from tqdm import tqdm

import datasets
from src.conf import MainConfig
from src.flow_helper import FlowHelper
from src.net import ViTDenoiser
from src.supplemental_net import ConvNextV2Loss, LPIPSLoss


def _lerp(a: float, b: float, p: float) -> float:
    return (b - a) * p + a


def _find_closest_factors(b: int) -> Tuple[int, int]:
    """Find the two closest integer factors of b."""
    start = int(math.sqrt(b))
    for nh in range(start, 0, -1):
        if b % nh == 0:
            nw = b // nh
            return nh, nw
    raise ValueError(f"Could not find factors for {b}")


def clear_cuda_cache(func):
    """
    A decorator that performs garbage collection and clears the CUDA cache
    before and after the decorated function is executed.
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        gc.collect()
        torch.cuda.empty_cache()

        result = func(*args, **kwargs)

        gc.collect()
        torch.cuda.empty_cache()

        return result

    return wrapper


def pil_to_tensor(x) -> torch.Tensor:
    """Convert PIL image to HWC uint8 tensor."""
    x = x.convert("RGB")
    w, h = x.size
    x_bytes = x.tobytes()
    tensor = torch.frombuffer(x_bytes, dtype=torch.uint8).reshape(h, w, 3)
    return tensor


class Trainer:
    def __init__(self, conf: MainConfig):
        self.conf = conf
        self.global_step = 0
        self.input_shape: Optional[Tuple[int, ...]] = None

        self._setup_paths()
        self._setup_data()
        self._setup_models()
        self._setup_optimizers()

        if conf.should_compile:
            self._compute_losses = torch.compile(
                self._compute_losses, fullgraph=True, dynamic=False
            )

        self._save_config()

    def _setup_paths(self) -> None:
        """Initialize output directory structure."""
        output_path = Path("out/")
        output_path.mkdir(exist_ok=True)

        # Find next run number
        existing = [p for p in output_path.iterdir() if p.is_dir() and p.name.isdigit()]
        run_num = max([int(p.name) for p in existing], default=-1) + 1

        self.run_path = output_path / f"{run_num:05d}"
        self.run_path.mkdir(parents=True)

        self.artifact_path = self.run_path / "artifacts"
        self.artifact_path.mkdir(exist_ok=True)

        self.checkpoint_path = self.run_path / "checkpoints"
        self.checkpoint_path.mkdir(exist_ok=True)

        self.log_path = (
            self.run_path / "logs.jsonl"
        )  # Fixed: was "log.jsonl" in read code
        self.wandb_run = wandb.init(
            project=self.conf.wandb_project_name, config=asdict(self.conf)
        )

    def _save_config(self) -> None:
        """Save run configuration to JSON."""
        config_path = self.run_path / "run_config.json"
        with open(config_path, "w") as f:
            json.dump(asdict(self.conf), f, indent=2)

    def _setup_data(self) -> None:
        """Initialize datasets and dataloaders."""
        dataset = datasets.load_dataset(self.conf.dataset_path_or_url)

        def apply_transforms(examples: dict[str, Any]) -> dict[str, Any]:
            images = examples.pop("image")
            examples["pixel_values"] = [pil_to_tensor(img) for img in images]
            return examples

        train_dataset = dataset["train"].with_transform(apply_transforms)
        test_dataset = dataset["validation"].with_transform(apply_transforms)

        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.conf.batch_size,
            shuffle=True,
            num_workers=self.conf.num_workers,
            drop_last=True,
            persistent_workers=True,
        )

        self.test_loader = DataLoader(
            test_dataset,
            batch_size=self.conf.batch_size,
            shuffle=False,
            num_workers=self.conf.num_workers,
        )

    def _setup_models(self) -> None:
        """Initialize models, EMA, and loss functions."""
        self.model = ViTDenoiser(self.conf.model).to(self.conf.device)

        self.ema_model = ViTDenoiser(self.conf.model).to(self.conf.device)
        self.ema_model.load_state_dict(self.model.state_dict())
        self.ema_model.requires_grad_(False)

        self.lpips_loss_fn = None
        if self.conf.lpips_weight > 0:
            self.lpips_loss_fn = LPIPSLoss().to(self.conf.device, self.conf.dtype)

        self.convnext_loss_fn = None
        if self.conf.convnext_weight > 0:
            self.convnext_loss_fn = ConvNextV2Loss().to(
                self.conf.device, self.conf.dtype
            )

        self.flow_helper = FlowHelper(self.conf.flow)

    def _classify_params(self) -> Tuple[list, list]:
        """Separate parameters for AdamW and Muon optimizers."""
        nonhidden_weights = []
        biases = []
        hidden_weights = []

        for name, parameter in self.model.named_parameters():
            is_hidden = "blocks." in name
            is_2d = parameter.ndim == 2

            if not is_2d:
                biases.append(parameter)
            elif is_hidden:
                hidden_weights.append(parameter)
            else:
                nonhidden_weights.append(parameter)

        adamw_groups = [
            {"params": nonhidden_weights, "use_weight_decay": True},
            {"params": biases, "use_weight_decay": False},
        ]
        muon_groups = [
            {"params": hidden_weights, "use_weight_decay": True},
        ]

        return adamw_groups, muon_groups

    def _setup_optimizers(self) -> None:
        """Initialize AdamW and Muon optimizers."""
        adamw_groups, muon_groups = self._classify_params()
        self.adamw_groups = adamw_groups  # Store for grad clipping
        self.muon_groups = muon_groups

        self.optim_muon = torch.optim.Muon(muon_groups, adjust_lr_fn="match_rms_adamw")
        self.optim_adamw = torch.optim.AdamW(adamw_groups)

    def _get_warmup_factor(self) -> float:
        """Calculate learning rate warmup factor."""
        return _lerp(0.1, 1.0, min(self.global_step / self.conf.num_warmup_steps, 1.0))

    def _update_ema(self, ema_beta: float) -> None:
        """Update EMA model parameters."""
        # ema_beta is the decay factor (close to 1.0)
        with torch.no_grad():
            for ema_p, p in zip(self.ema_model.parameters(), self.model.parameters()):
                ema_p.lerp_(p, 1.0 - ema_beta)

    def _forward_model(
        self,
        model: ViTDenoiser,
        x_t: torch.Tensor,
        r: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass through the model handling patching/unpatching."""
        b, h, w, c = x_t.shape
        device, dtype = x_t.device, x_t.dtype
        nph = h // self.conf.patch_size
        npw = w // self.conf.patch_size

        # Create patch coordinates
        h_coords = torch.linspace(-1.0, 1.0, nph, device=device)
        w_coords = torch.linspace(-1.0, 1.0, npw, device=device)
        patch_coords = torch.stack(
            torch.meshgrid(h_coords, w_coords, indexing="ij"), -1
        )
        patch_coords = repeat(patch_coords, "nph npw nd -> b (nph npw) nd", b=b)

        # Patchify input
        patches = rearrange(
            x_t,
            "b (nph ph) (npw pw) c -> b (nph npw) (ph pw c)",
            ph=self.conf.patch_size,
            pw=self.conf.patch_size,
        )

        # Prepare timesteps
        r = r.squeeze().unsqueeze(-1)
        t = t.squeeze().unsqueeze(-1)

        # Forward
        with torch.autocast(
            self.conf.device.type,
            self.conf.dtype,
            enabled=patches.dtype != torch.float32,
        ):
            output = model(
                patches=patches,
                terminal_timesteps=r,
                timesteps=t,
                patch_coords=patch_coords,
            )

        # Unpatchify output
        x_0_hat = output.prediction
        x_0_hat = rearrange(
            x_0_hat,
            "b (nph npw) (ph pw c) -> b (nph ph) (npw pw) c",
            nph=nph,
            npw=npw,
            c=3,
            ph=self.conf.patch_size,
            pw=self.conf.patch_size,
        )

        return x_0_hat.float()

    def _compute_losses(self, batch: dict[str, Any]):
        pixel_values = batch.pop("pixel_values")
        pixel_values = (
            pixel_values.to(self.conf.device, torch.float32)
            .div_(255.0)
            .mul_(2.0)
            .sub_(1.0)
        )

        # Store input shape for validation
        self.input_shape = pixel_values.shape

        loss_dict, extra_dict = self.flow_helper.compute_meanflow_loss(
            x_0=pixel_values,
            net=partial(self._forward_model, self.model),
            timesteps_shape=(pixel_values.shape[0], 1, 1, 1),
        )

        x_0_hat = extra_dict["x_0_hat"]
        timesteps = extra_dict["t"]
        x_0_hat = rearrange(x_0_hat, "b h w c -> b c h w")
        pixel_values = rearrange(pixel_values, "b h w c -> b c h w")

        with torch.autocast(
            self.conf.device.type,
            self.conf.dtype,
            enabled=self.conf.dtype != torch.float32,
        ):
            # indices of the least noisy samples
            indices = timesteps.squeeze().argsort(descending=False)
            indices = indices[
                : int(indices.shape[0] * self.conf.perceptual_loss_proportion)
            ]

            x_0_hat = x_0_hat[indices]
            pixel_values = pixel_values[indices]

            lpips_loss = 0.0
            if self.conf.lpips_weight > 0.0 and self.lpips_loss_fn is not None:
                lpips_loss = self.lpips_loss_fn(x_0_hat, pixel_values)

            convnext_loss = 0.0
            if self.conf.convnext_weight > 0.0 and self.convnext_loss_fn is not None:
                convnext_loss = self.convnext_loss_fn(x_0_hat, pixel_values)

        total_loss = (
            loss_dict["loss"]
            + lpips_loss * self.conf.lpips_weight
            + convnext_loss * self.conf.convnext_weight
        )
        loss_dict["total_loss"] = total_loss
        loss_dict["lpips_loss"] = lpips_loss
        loss_dict["convnext_loss"] = convnext_loss

        return loss_dict

    def _train_step(self, batch: dict[str, Any]) -> dict[str, float]:
        """Execute a single training step."""
        loss_dict = self._compute_losses(batch)
        del batch

        total_loss = loss_dict["total_loss"]
        total_loss.backward()

        loss_dict = {
            k: v.detach().cpu().item() if isinstance(v, torch.Tensor) else v
            for k, v in loss_dict.items()
        }

        # Gradient clipping (per param group to preserve original behavior)
        for p_group in self.adamw_groups + self.muon_groups:
            torch.nn.utils.clip_grad_norm_(p_group["params"], max_norm=1.0)

        # Update learning rates with warmup
        warmup_p = self._get_warmup_factor()

        lr_muon = self.conf.lr_muon * warmup_p
        for p in self.optim_muon.param_groups:
            p["lr"] = lr_muon
            p["weight_decay"] = self.conf.weight_decay_muon
            p["momentum"] = self.conf.momentum_muon

        lr_adamw = self.conf.lr_adamw * warmup_p
        for p in self.optim_adamw.param_groups:
            p["lr"] = lr_adamw
            p["weight_decay"] = (
                self.conf.weight_decay_adamw if p["use_weight_decay"] else 0.0
            )
            p["betas"] = self.conf.betas_adamw

        # Optimizer steps
        self.optim_adamw.step()
        self.optim_muon.step()
        self.optim_adamw.zero_grad(set_to_none=True)
        self.optim_muon.zero_grad(set_to_none=True)

        # Update EMA (with warmup on the decay factor)
        ema_beta = (
            warmup_p * self.conf.ema_beta
        )  # Starts low, increases to conf.ema_beta
        self._update_ema(ema_beta)

        log_dict = {
            **loss_dict,
            "lr_adamw": lr_adamw,
            "lr_muon": lr_muon,
        }

        return log_dict

    @clear_cuda_cache
    @torch.inference_mode()
    def _validate_and_log(self) -> None:
        """Generate validation samples and save images."""
        if self.input_shape is None:
            return

        torch_rng = torch.Generator(self.conf.device)
        x_1 = torch.randn(
            *self.input_shape,
            device=self.conf.device,
            dtype=self.conf.dtype,
            generator=torch_rng,
        )

        with torch.autocast(self.conf.device.type, self.conf.dtype):
            # Single step inference
            x_0 = self.flow_helper.sample_euler(
                partial(self._forward_model, self.ema_model),
                x_1=x_1,
                num_steps=1,
            )

        # Convert to uint8 HWC
        x_0 = (
            x_0.float().add(1).div(2).clamp(0, 1).mul(255).round().to(torch.uint8).cpu()
        )

        # Add grid padding
        pad = 2
        h_padding = torch.zeros(
            x_0.shape[0], pad, x_0.shape[2], x_0.shape[3], dtype=torch.uint8
        )
        x_0 = torch.cat((x_0, h_padding), dim=1)
        w_padding = torch.zeros(
            x_0.shape[0], x_0.shape[1], pad, x_0.shape[3], dtype=torch.uint8
        )
        x_0 = torch.cat((x_0, w_padding), dim=2)

        # Arrange in grid
        b = self.input_shape[0]
        nh, nw = _find_closest_factors(b)
        grid = rearrange(x_0, "(nh nw) h w c -> c (nh h) (nw w)", nh=nh, nw=nw)

        # Remove padding from final grid
        grid = grid[:, :-pad, :-pad]

        save_path = self.artifact_path / f"{self.global_step:06d}.png"
        torchvision.io.write_png(grid, str(save_path))

    def _save_checkpoint(self) -> None:
        """Save model checkpoint and rotate old checkpoints."""
        checkpoint = {
            "model": self.model.state_dict(),
            "ema_model": self.ema_model.state_dict(),
            "adamw": self.optim_adamw.state_dict(),
            "muon": self.optim_muon.state_dict(),
            "global_step": self.global_step,
            "wandb_run_id": self.wandb_run.id,
        }

        save_path = self.checkpoint_path / f"{self.global_step:06d}.pt"
        torch.save(checkpoint, save_path)

        # Rotate checkpoints: keep only the most recent N
        max_checkpoints = self.conf.max_num_checkpoints
        all_checkpoints = sorted(self.checkpoint_path.glob("*.pt"))
        while len(all_checkpoints) > max_checkpoints:
            old_ckpt = all_checkpoints.pop(0)
            old_ckpt.unlink()

    def _maybe_plot_losses(self) -> None:
        """Generate and save loss curve plot."""
        if not self.log_path.exists():
            return

        df = pd.read_json(self.log_path, lines=True)

        if len(df) < 2:
            return

        df["total_loss_smoothed"] = (
            df["total_loss"].ewm(adjust=False, alpha=0.005).mean()
        )

        plt.figure(figsize=(10, 6))
        plt.plot(df["global_step"], df["total_loss_smoothed"], label="Smoothed")
        plt.plot(df["global_step"], df["total_loss"], alpha=0.1, label="Raw")
        plt.ylim(df["total_loss"].min(), df["total_loss_smoothed"].quantile(0.9))
        plt.xlabel("Global Step")
        plt.ylabel("Total Loss")
        plt.legend()
        plt.savefig(self.run_path / "total_loss.png")
        plt.close()

    def train(self) -> None:
        """Main training loop."""
        for epoch in range(self.conf.num_train_epochs):
            for batch in tqdm(self.train_loader, desc=f"Epoch {epoch}"):
                log_dict = self._train_step(batch)

                # Convert tensors to scalars for logging
                log_dict["epoch"] = epoch
                log_dict["global_step"] = self.global_step

                should_validate = (
                    self.global_step % self.conf.validate_every_num_steps == 0
                )
                if should_validate:
                    self._validate_and_log()
                    self._maybe_plot_losses()

                should_save = self.global_step % self.conf.save_every_num_steps == 0
                if should_save:
                    self._save_checkpoint()

                # JSONL logging (append mode)
                with open(self.log_path, "a") as f:
                    f.write(json.dumps(log_dict) + "\n")

                should_log_wandb = (
                    self.global_step % self.conf.wandb_log_every_num_steps == 0
                ) or should_validate
                if should_log_wandb and self.wandb_run:
                    wandb.log(log_dict, step=self.global_step)

                self.global_step += 1

        if self.wandb_run:
            self.wandb_run.finish()

        self._save_checkpoint()
