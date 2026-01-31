from dataclasses import dataclass
import math
from typing import Callable
import torch
import torch.nn.functional as F


@dataclass
class FlowHelperConfig:
    time_shift_dim: float = 12288
    time_shift_base: float = 4096
    eps: float = 0.05


class FlowHelper:
    """
    Flow from noise (t=1.0) to data (t=0.0)

    using a x-prediction network
    and shifted timesteps
    """

    def __init__(self, conf: FlowHelperConfig = FlowHelperConfig()):
        self.conf = conf

    def _get_timestep_interval(self):
        t_0 = 0.0
        t_1 = 1 - 1 / 1000
        return t_0, t_1

    def shift_timesteps(self, t: torch.Tensor):
        shift = math.sqrt(self.conf.time_shift_dim / self.conf.time_shift_base)
        # Shifts timesteps to be larger (assuming shift > 1)
        t = 1 - (1 - t) / (1 - t + shift * t)
        return t

    def compute_flow_matching_loss(
        self,
        x_0: torch.Tensor,
        # (x_t, r, t) -> x_0_hat
        net: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
        timesteps_shape: tuple,
        torch_rng: torch.Generator | None = None,
    ):
        device = x_0.device

        t = torch.rand(*timesteps_shape, generator=torch_rng, device=device)
        t = self.shift_timesteps(t)
        x_1 = torch.randn_like(x_0)
        z = (1 - t) * x_0 + t * x_1
        r = torch.zeros_like(t)
        v = (z - net(z, r, t)) / t.clip(self.conf.eps)
        fm_loss = F.mse_loss(v, x_1 - x_0)
        return {"loss": fm_loss}, {}

    def compute_meanflow_loss(
        self,
        x_0: torch.Tensor,
        # (x_t, r, t) -> x_0_hat
        net: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
        timesteps_shape: tuple,
        torch_rng: torch.Generator | None = None,
    ):
        device = x_0.device

        tr = torch.rand(*timesteps_shape, 2, generator=torch_rng, device=device)
        # Shift timesteps to be larger (more noisy)
        tr = self.shift_timesteps(tr)
        t = tr.amax(-1)
        r = tr.amin(-1)

        x_1 = torch.randn_like(x_0)
        z = (1 - t) * x_0 + t * x_1

        def u_fn(z_in, r_in, t_in):
            return (z_in - net(z_in, r_in, t_in)) / t_in.clip(self.conf.eps)

        # Predict the instantaneous velocity
        v = u_fn(z, t, t)

        # Mean flow
        u, dudt = torch.func.jvp(
            u_fn,
            (z, r, t),
            (v, torch.zeros_like(r), torch.ones_like(t)),
        )

        V = u + (t - r) * dudt.detach()
        loss_mf = F.mse_loss(V, x_1 - x_0)

        return (
            {
                "loss": loss_mf,
            },
            {"x_0_hat", None},
        )
