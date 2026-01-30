from dataclasses import dataclass, field

import torch

from src.flow_helper import FlowHelperConfig
from src.net import ViTDenoiserConfig


@dataclass
class MainConfig:
    model: ViTDenoiserConfig = field(default_factory=ViTDenoiserConfig)

    flow: FlowHelperConfig = field(default_factory=FlowHelperConfig)

    dataset_path_or_url: str = "benjamin-paine/imagenet-1k-64x64"
    patch_size: int = 4

    batch_size: int = 8
    num_workers: int = 0

    num_train_epochs: int = 1000

    ema_beta: float = 0.9995

    num_warmup_steps: int = 10000
    lr_muon: float = 1e-4
    lr_adamw: float = 5e-5
    betas_adamw: tuple[float, float] = (0.9, 0.999)
    weight_decay_adamw: float = 1e-2
    weight_decay_muon: float = 1e-1
    momentum_muon: float = 0.99

    lpips_weight: float = 0.4
    convnext_weight: float = 0.1

    wandb_project_name: str = "minimal-pixel-meanflow"
    wandb_log_every_num_steps: int = 50
    validate_every_num_steps: int = 1000

    device_str: str = "cuda"
    dtype_str: str = "bfloat16"

    def __post_init__(self):
        self.device = torch.device(self.device_str)
        self.dtype = getattr(torch, self.dtype_str)
