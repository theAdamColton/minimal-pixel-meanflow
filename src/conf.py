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

    batch_size: int = 64
    num_workers: int = 4

    num_train_epochs: int = 1000

    ema_beta: float = 0.999

    num_warmup_steps: int = 40000
    lr_muon: float = 5e-3
    lr_adamw: float = 1e-3
    betas_adamw: tuple[float, float] = (0.9, 0.997)
    weight_decay_adamw: float = 1e-2
    weight_decay_muon: float = 1e-2
    momentum_muon: float = 0.95

    lpips_weight: float = 0.0
    convnext_weight: float = 0.0

    wandb_project_name: str = "minimal-pixel-meanflow"
    wandb_log_every_num_steps: int = 50
    validate_every_num_steps: int = 200

    should_compile: bool = False

    device_str: str = "cuda"
    dtype_str: str = "bfloat16"

    def __post_init__(self):
        self.device = torch.device(self.device_str)
        self.dtype = getattr(torch, self.dtype_str)
