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
import torch.nn.functional as F
from tqdm import tqdm

import datasets
from src.conf import MainConfig
from src.flow_helper import FlowHelper
from src.net import ViTDenoiser, ViTDenoiserOutput
from src.supplemental_net import (
    ConvNextV2Loss,
    LPIPSLoss,
    DinoV3Encoder,
    REPAProjector2D,
)


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


def _make_grid(x: torch.Tensor):
    """
    x: shape (b,h,w,c) containing pixel values
    """
    b, h, w, c = x.shape

    if x.dtype.is_floating_point:
        # Convert [-1,1] pixel values to uint8
        x = x.float().add(1).div(2).clamp(0, 1).mul(255).round().to(torch.uint8)
    else:
        assert x.dtype == torch.uint8

    x = x.cpu()

    # Add grid padding
    pad = 2
    h_padding = torch.zeros(b, pad, w, c, dtype=torch.uint8)
    x = torch.cat((x, h_padding), dim=1)
    w_padding = torch.zeros(b, h + pad, pad, c, dtype=torch.uint8)
    x = torch.cat((x, w_padding), dim=2)

    # Arrange in grid
    nh, nw = _find_closest_factors(b)
    grid = rearrange(x, "(nh nw) h w c -> c (nh h) (nw w)", nh=nh, nw=nw)

    # Remove padding from final grid
    grid = grid[:, :-pad, :-pad]

    return grid


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
        self.num_trained_samples = 0
        self.input_shape: Optional[Tuple[int, ...]] = None

        self.wandb_run_id = None
        self._setup_paths()
        self._setup_data()
        self._setup_models()
        self._setup_optimizers()

        self._compute_losses_training = self._compute_losses
        if conf.should_compile:
            self._compute_losses_training = torch.compile(
                self._compute_losses, fullgraph=True, dynamic=False
            )

        self._save_config()

        if conf.resume_checkpoint_path is not None:
            d = torch.load(conf.resume_checkpoint_path)
            print("Loading checkpoint")
            self.model.load_state_dict(d["model"])
            self.ema_model.load_state_dict(d["ema_model"])
            self.repa_projector.load_state_dict(d["repa_projector"])
            self.optim_adamw.load_state_dict(d["adamw"])
            self.optim_muon.load_state_dict(d["muon"])
            self.global_step = d["global_step"]
            self.num_trained_samples = d["num_trained_samples"]
            self.wandb_run_id = d["wandb_run_id"]

        wandb_run = wandb.init(
            project=self.conf.wandb_project_name,
            config=asdict(self.conf),
            id=self.wandb_run_id if not conf.wandb_force_new_run else None,
        )
        self.wandb_run_id = wandb_run.id

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

        self.log_path = self.run_path / "logs.jsonl"

    def _save_config(self) -> None:
        """Save run configuration to JSON."""
        config_path = self.run_path / "run_config.json"
        with open(config_path, "w") as f:
            json.dump(asdict(self.conf), f, indent=2)

    def _setup_data(self) -> None:
        """Initialize datasets and dataloaders."""
        dataset = datasets.load_dataset(self.conf.dataset_path_or_url)

        def transform_img(img):
            return pil_to_tensor(img)

        def transform_row(row):
            row[self.conf.dataset_image_column_name] = transform_img(
                row[self.conf.dataset_image_column_name]
            )
            return row

        def transform_batch(samples):
            samples[self.conf.dataset_image_column_name] = [
                transform_img(image)
                for image in samples[self.conf.dataset_image_column_name]
            ]
            return samples

        self.train_dataset_length = len(dataset["train"])

        train_dataset = (
            dataset["train"]
            .shuffle(seed=42)
            .flatten_indices()
            .to_iterable_dataset(num_shards=1024)
            .repeat(None)
            .shuffle(buffer_size=4096)
            .map(transform_row)
        )

        test_dataset = dataset["validation"].with_transform(transform_batch)

        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.conf.batch_size,
            num_workers=self.conf.num_workers,
            drop_last=True,
            persistent_workers=True,
        )

        self.test_loader = DataLoader(
            test_dataset, batch_size=self.conf.batch_size, shuffle=False
        )

    def _setup_models(self) -> None:
        """Initialize models, EMA, and loss functions."""

        self.model = ViTDenoiser(self.conf.model).to(self.conf.device)
        self.repa_projector = REPAProjector2D(
            in_channels=self.conf.model.hidden_size,
            out_channels=self.conf.repa_output_size,
        ).to(self.conf.device)

        self.ema_model = ViTDenoiser(self.conf.model).to(self.conf.device)
        self.ema_model.load_state_dict(self.model.state_dict())
        self.ema_model.requires_grad_(False)

        self.dinov3_encoder = None
        if self.conf.repa_weight > 0:
            self.dinov3_encoder = DinoV3Encoder().to(self.conf.device, self.conf.dtype)

        self.lpips_loss_fn = None
        if self.conf.lpips_weight > 0:
            self.lpips_loss_fn = LPIPSLoss().to(self.conf.device, self.conf.dtype)

        self.convnext_loss_fn = None
        if self.conf.convnext_weight > 0:
            self.convnext_loss_fn = ConvNextV2Loss().to(
                self.conf.device, self.conf.dtype
            )

        self.flow_helper = FlowHelper(self.conf.flow)

    def _log(self, log_dict):
        with open(self.log_path, "a") as f:
            f.write(json.dumps(log_dict) + "\n")

    def _log_wandb(self, log_dict):
        wandb.log(log_dict, step=self.global_step)

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

        nonhidden_weights.append(self.repa_projector.conv.weight)
        if self.repa_projector.conv.bias is not None:
            biases.append(self.repa_projector.conv.bias)

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

        self.adamw_groups = adamw_groups
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
                ema_p.lerp_(p.to(ema_p.dtype), 1.0 - ema_beta)

    def _autocast(self):
        return torch.autocast(
            self.conf.device.type,
            self.conf.dtype,
            enabled=self.conf.dtype != torch.float32,
        )

    def _forward_pixel_values(
        self,
        model: ViTDenoiser,
        x_t: torch.Tensor,
        r: torch.Tensor,
        t: torch.Tensor,
        cfg: torch.Tensor,
        labels: torch.Tensor,
        return_layer_indices=None,
    ):
        """Forward pass through the model handling patching/unpatching."""
        b, h, w, _ = x_t.shape
        device, dtype = x_t.device, x_t.dtype
        nph = h // self.conf.patch_size
        npw = w // self.conf.patch_size

        # Create patch coordinates
        h_coords = torch.linspace(-1.0, 1.0, nph, device=device, dtype=dtype)
        w_coords = torch.linspace(-1.0, 1.0, npw, device=device, dtype=dtype)
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

        r = r.view(-1, 1)
        t = t.view(-1, 1)
        cfg = cfg.view(-1, 1)
        labels = labels.view(-1, 1)

        # Forward
        output: ViTDenoiserOutput = model(
            patches=patches,
            terminal_timesteps=r,
            timesteps=t,
            patch_coords=patch_coords,
            cfg=cfg,
            class_ids=labels,
            return_layer_indices=return_layer_indices,
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

        x_0_hat = x_0_hat.to(dtype)

        return x_0_hat, {"layer_hidden_states": output.layer_hidden_states}

    def _compute_repa_loss(
        self, pixel_values: torch.Tensor, student_hidden_states: torch.Tensor
    ):
        _, h, w, _ = pixel_values.shape

        with torch.no_grad():
            _, teacher_hidden_states = self.dinov3_encoder(pixel_values)

        teacher_hidden_states = rearrange(
            teacher_hidden_states,
            "b (nph npw) d -> b d nph npw",
            nph=h // self.dinov3_encoder.patch_size,
            npw=w // self.dinov3_encoder.patch_size,
        )

        # Spatial normalization
        mean = teacher_hidden_states.mean(dim=(2, 3), keepdim=True)
        std = teacher_hidden_states.std(dim=(2, 3), keepdim=True)
        teacher_hidden_states = (teacher_hidden_states - mean) / std

        student_nph = h // self.conf.patch_size
        student_npw = w // self.conf.patch_size
        student_hidden_states = rearrange(
            student_hidden_states,
            "b (nph npw) d -> b d nph npw",
            nph=student_nph,
            npw=student_npw,
        )

        # Handle differing patch sizes between model and dinov3
        scale_factor = self.conf.patch_size / self.dinov3_encoder.patch_size
        student_hidden_states = F.interpolate(
            student_hidden_states,
            scale_factor=scale_factor,
            mode="bilinear",
        )

        student_hidden_states = self.repa_projector(student_hidden_states)
        repa_loss = -F.cosine_similarity(
            student_hidden_states, teacher_hidden_states, dim=1
        ).mean()

        return repa_loss, {
            "projected_hidden_states": student_hidden_states,
            "teacher_hidden_states": teacher_hidden_states,
        }

    def _compute_losses(
        self, batch: dict[str, Any], torch_rng: torch.Generator | None = None
    ):
        device = self.conf.device
        dtype = self.conf.dtype

        pixel_values = batch.pop(self.conf.dataset_image_column_name)
        labels = batch.pop(self.conf.dataset_label_column_name)

        pixel_values = (
            pixel_values.to(device, torch.float32).div_(255.0).mul_(2.0).sub_(1.0)
        )
        labels = labels.to(device)

        # Store input shape for validation
        self.input_shape = pixel_values.shape

        extra_dict = dict()

        loss_dict, meanflow_extra_dict = self.flow_helper.compute_meanflow_loss(
            x_0=pixel_values,
            labels=labels,
            net=partial(
                self._forward_pixel_values,
                self.model,
                return_layer_indices=[self.conf.repa_depth],
            ),
            timesteps_shape=(pixel_values.shape[0], 1, 1, 1),
            unc_label_id=self.conf.model.unconditional_class_id,
            torch_rng=torch_rng,
        )

        x_0_hat = meanflow_extra_dict["x_0_hat"]
        timesteps = meanflow_extra_dict["t"]

        repa_loss = 0.0
        if self.conf.repa_weight > 0.0 and self.dinov3_encoder is not None:
            # Hidden states extracted from the nth layer when computing the
            # instantaneous velocity
            student_hidden_states = meanflow_extra_dict["layer_hidden_states"].squeeze(
                0
            )
            repa_loss, extra_dict_repa = self._compute_repa_loss(
                pixel_values, student_hidden_states
            )
            extra_dict.update(extra_dict_repa)

        x_0_hat = rearrange(x_0_hat, "b h w c -> b c h w")
        pixel_values = rearrange(pixel_values, "b h w c -> b c h w")

        # indices of the least noisy samples
        indices = timesteps.squeeze().argsort(descending=False)
        indices = indices[
            : int(indices.shape[0] * self.conf.perceptual_loss_proportion)
        ]

        # Take only the least noisy samples
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
            + repa_loss * self.conf.repa_weight
        )
        loss_dict["repa_loss"] = repa_loss
        loss_dict["total_loss"] = total_loss
        loss_dict["lpips_loss"] = lpips_loss
        loss_dict["convnext_loss"] = convnext_loss

        extra_dict["x_0_hat"] = x_0_hat
        extra_dict["x_0"] = pixel_values

        return loss_dict, extra_dict

    def _train_step(self, batch: dict[str, Any]) -> dict[str, float]:
        """Execute a single training step."""
        with self._autocast():
            loss_dict, _ = self._compute_losses_training(batch)

        total_loss = loss_dict["total_loss"]
        total_loss.backward()

        loss_dict = {
            k: v.detach().cpu().item() if isinstance(v, torch.Tensor) else v
            for k, v in loss_dict.items()
        }

        # Gradient clipping
        for p_group in self.adamw_groups + self.muon_groups:
            torch.nn.utils.clip_grad_norm_(
                p_group["params"], max_norm=self.conf.max_gradient_norm
            )

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

        # Update EMA with warmup
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
    def _fast_validate(self):
        model = self.ema_model
        device = self.conf.device
        dtype = torch.float32

        assert self.input_shape is not None

        # Generate some samples
        torch_rng = torch.Generator(device)
        x_1 = torch.randn(
            *self.input_shape,
            device=device,
            dtype=dtype,
            generator=torch_rng,
        )
        batch_size = self.input_shape[0]
        labels = torch.arange(batch_size, device=device)
        cfg = torch.full(
            (batch_size,), self.conf.validation_cfg, device=device, dtype=dtype
        )

        with self._autocast():
            # Single step inference
            x_0 = self.flow_helper.sample_euler(
                partial(self._forward_pixel_values, model),
                x_1=x_1,
                labels=labels,
                cfg=cfg,
                num_steps=1,
            )

        grid = _make_grid(x_0)
        save_path = self.artifact_path / f"{self.global_step:07d}.png"
        torchvision.io.write_png(grid, str(save_path))

        # Compute losses on validation batches
        all_val_losses = []
        for batch in self.test_loader:
            with self._autocast():
                loss_dict, extra_dict = self._compute_losses(batch, torch_rng=torch_rng)
            loss_dict = {
                k: v.detach().cpu().float() if isinstance(v, torch.Tensor) else v
                for k, v in loss_dict.items()
            }
            all_val_losses.append(loss_dict)
            if len(all_val_losses) >= self.conf.num_validation_batches:
                break

        # Save reconstruction grid
        x_0_hat = extra_dict["x_0_hat"]
        x_0 = extra_dict["x_0"]
        vis = torch.cat((x_0, x_0_hat), -1)
        grid = _make_grid(vis.movedim(1, -1))
        save_path = self.artifact_path / f"noisy_rec_{self.global_step:07d}.png"
        torchvision.io.write_png(grid, str(save_path))

        # Save feature visualization
        features = extra_dict.get("projected_hidden_states")
        if features is not None:
            _, _, nph, npw = features.shape
            features = rearrange(features, "b d nph npw -> b (nph npw) d")
            features_rgb, _, _ = torch.pca_lowrank(features.float(), q=3, niter=10)
            min = features_rgb.amin(dim=1, keepdim=True)
            max = features_rgb.amax(dim=1, keepdim=True)
            features_rgb = (features_rgb - min) / (max - min).clip(1e-5)
            features_rgb = (
                features_rgb.clip(0, 1).mul(255).round().to(torch.uint8).cpu()
            )
            features_rgb = repeat(
                features_rgb,
                "b (nph npw) c -> b (nph ph) (npw pw) c",
                nph=nph,
                npw=npw,
                ph=self.dinov3_encoder.patch_size,
                pw=self.dinov3_encoder.patch_size,
            )
            grid = _make_grid(features_rgb)
            save_path = self.artifact_path / f"features_{self.global_step:07d}.png"
            torchvision.io.write_png(grid, str(save_path))

        # Aggregate val losses
        log_dict = dict()
        for k in all_val_losses[0].keys():
            v = [row[k] for row in all_val_losses]
            v = [x.item() if isinstance(x, torch.Tensor) else x for x in v]
            v_mean = sum(v) / len(v)
            log_dict[f"val_{k}"] = v_mean

        return log_dict

    def _save_checkpoint(self) -> None:
        """Save model checkpoint and rotate old checkpoints."""
        checkpoint = {
            "model": self.model.state_dict(),
            "ema_model": self.ema_model.state_dict(),
            "repa_projector": self.repa_projector.state_dict(),
            "adamw": self.optim_adamw.state_dict(),
            "muon": self.optim_muon.state_dict(),
            "global_step": self.global_step,
            "num_trained_samples": self.num_trained_samples,
            "wandb_run_id": self.wandb_run_id,
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
        prog_bar = tqdm(dynamic_ncols=True, leave=True)
        data_iter = iter(self.train_loader)
        while True:
            batch = next(data_iter)
            num_batch_samples = len(batch[self.conf.dataset_image_column_name])
            log_dict = self._train_step(batch)

            log_dict["epoch"] = self.num_trained_samples / self.train_dataset_length
            log_dict["global_step"] = self.global_step

            should_validate = self.global_step % self.conf.validate_every_num_steps == 0
            if should_validate:
                val_log_dict = self._fast_validate()
                log_dict.update(val_log_dict)

            should_log_wandb = (
                self.global_step % self.conf.wandb_log_every_num_steps == 0
            ) or should_validate
            if should_log_wandb:
                self._log_wandb(log_dict)

            self._log(log_dict)

            if should_validate:
                self._maybe_plot_losses()

            prog_bar.update(1)
            prog_bar.set_description(
                f"epoch:{log_dict['epoch']:.2f} step:{log_dict['global_step']:07d} loss:{log_dict['total_loss']:.3f}"
            )
            self.global_step += 1
            self.num_trained_samples += num_batch_samples

            should_save = self.global_step % self.conf.save_every_num_steps == 0
            if should_save:
                self._save_checkpoint()

            if log_dict["epoch"] >= self.conf.num_train_epochs:
                break

        prog_bar.close()
        self._save_checkpoint()
