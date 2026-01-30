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

        # Clip t for numerical stability in division
        t_clipped = t.clip(self.conf.eps)

        # ------------------------------------------------------------------
        # 1. Compute u (Average Velocity) at (z, r, t)
        # We need gradients here for the loss, so this stays outside no_grad.
        # ------------------------------------------------------------------
        x_0_hat_r = net(z, r, t)
        u = (z - x_0_hat_r) / t_clipped

        # ------------------------------------------------------------------
        # 2. Compute dudt (Correction Term)
        # Your algo says: V = u + (t-r) * stopgrad(dudt)
        # We use no_grad() to implement stopgrad AND fix the autocast crash.
        # ------------------------------------------------------------------
        with torch.no_grad():
            # Calculate v: Instantaneous velocity at (z, t, t)
            # Note: We pass 't' as the second argument (r=t) per your algo
            x_0_hat_t = net(z, t, t)
            v = (z - x_0_hat_t) / t_clipped

            # Define the function for JVP
            def u_fn_jvp(z_in, r_in, t_in):
                return (z_in - net(z_in, r_in, t_in)) / t_in.clip(self.conf.eps)

            # Compute JVP
            # Primals: (z, r, t)
            # Tangents: (v, 0, 1) -> v matches z, 0 matches r, 1 matches t
            _, dudt = torch.func.jvp(
                u_fn_jvp, (z, r, t), (v, torch.zeros_like(r), torch.ones_like(t))
            )

        # ------------------------------------------------------------------
        # 3. Combine
        # ------------------------------------------------------------------
        # dudt is already detached because it came from the no_grad block
        V = u + (t - r) * dudt

        # Target is (e - x) which is equivalent to x_1 - x_0
        loss = F.mse_loss(V, x_1 - x_0)

        return {
            "loss": loss,
            "x_0_hat": x_0_hat_r,  # Log the main prediction
            "t": t,
        }
