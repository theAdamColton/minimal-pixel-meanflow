from dataclasses import dataclass
import math
from typing import Callable
import torch
import torch.nn.functional as F


@dataclass
class FlowHelperConfig:
    time_shift_dim: float = 4096
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

    def compute_meanflow_loss(
        self,
        x_0: torch.Tensor,
        # (x_t, r, t) -> x_0_hat
        net: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
        timesteps_shape: tuple,
        torch_rng: torch.Generator | None = None,
    ):
        device = x_0.device

        # U(0,1)
        t = torch.rand(timesteps_shape, generator=torch_rng, device=device)
        # U(0,t)
        r = torch.rand(timesteps_shape, generator=torch_rng, device=device) * t

        # Shift timesteps to be larger (more noisy)
        t = self.shift_timesteps(t)
        r = self.shift_timesteps(r)

        x_1 = torch.randn_like(x_0)

        z = (1 - t) * x_0 + t * x_1

        x_0_hat = net(z, r, t)
        # instantaneous velocity v at time t
        v = z - x_0_hat / t.clip(self.conf.eps)

        # average velocity u from x-prediction
        def u_fn(z_in, r_in, t_in):
            return (z_in - net(z_in, r_in, t_in)) / t_in.clip(self.conf.eps)

        # predict u and dudt
        u, dudt = torch.func.jvp(
            u_fn, (z, r, t), (v, torch.zeros_like(r), torch.ones_like(t))
        )

        # compound function V
        V = u + (t - r) * dudt.detach()

        loss_pmf = F.mse_loss(V, x_1 - x_0)

        return {
            "loss_pmf": loss_pmf,
            "x_0_hat": x_0_hat,
            "t": t,
        }
