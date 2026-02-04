from dataclasses import dataclass
import math
from typing import Callable, Literal, Tuple, Dict
import torch
import torch.nn.functional as F


def unsqueeze_trailing(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    while x.ndim < y.ndim:
        x = x.unsqueeze(-1)
    return x


@dataclass
class FlowHelperConfig:
    time_shift_dim: float = 12288
    time_shift_base: float = 4096
    eps: float = 1e-3
    prediction_mode: Literal["clean_input", "velocity"] = "clean_input"

    norm_p: float = 1.0
    norm_eps: float = 0.01


class FlowHelper:
    """
    Flow from noise (t=1.0) to data (t=0.0).
    """

    def __init__(self, conf: FlowHelperConfig = FlowHelperConfig()):
        self.conf = conf

    def _get_timestep_interval(self):
        t_0 = 0.0
        t_1 = 1.0 - 1e-5
        return t_0, t_1

    def shift_timesteps(self, t: torch.Tensor):
        """
        Shifts timesteps towards noise (1.0) based on resolution/dimension.
        """
        shift = math.sqrt(self.conf.time_shift_dim / self.conf.time_shift_base)
        t = 1 - (1 - t) / (1 - t + shift * t)
        return t

    def get_velocity(
        self, model_out: torch.Tensor, z: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        """
        Converts model output to velocity u based on prediction mode.
        If prediction is x_0, u = (z - x_0) / t.
        """
        if self.conf.prediction_mode == "clean_input":
            t_clipped = t.clamp(min=self.conf.eps)
            u = (z - model_out) / t_clipped
        elif self.conf.prediction_mode == "velocity":
            u = model_out
        else:
            raise ValueError(f"Unknown prediction mode: {self.conf.prediction_mode}")
        return u

    def adaptive_weighting(self, loss: torch.Tensor) -> torch.Tensor:
        adp_wt = (loss + self.conf.norm_eps).pow(self.conf.norm_p)
        return loss / adp_wt.detach()

    def sample_euler(
        self,
        net: Callable[
            [torch.Tensor, torch.Tensor, torch.Tensor], tuple[torch.Tensor, dict]
        ],
        x_1: torch.Tensor,
        num_steps: int = 4,
    ):
        device = x_1.device
        t_0, t_1 = self._get_timestep_interval()

        t = torch.linspace(t_1, t_0, num_steps + 1, device=device)

        x_t = x_1

        for i in range(num_steps):
            t_curr = t[i]
            t_next = t[i + 1]
            dt = t_next - t_curr

            t_in = unsqueeze_trailing(t_curr, x_t)
            r_in = unsqueeze_trailing(t_next, x_t)

            model_out, _ = net(x_t, r_in, t_in)
            u = self.get_velocity(model_out, x_t, t_in)

            x_t = x_t + u * dt

        return x_t

    def compute_meanflow_loss(
        self,
        x_0: torch.Tensor,
        # Net signature: (x_t, r, t) -> (Prediction (x_0 or v), dict)
        net: Callable[
            [torch.Tensor, torch.Tensor, torch.Tensor], tuple[torch.Tensor, dict]
        ],
        timesteps_shape: tuple,
        torch_rng: torch.Generator | None = None,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        device, dtype = x_0.device, x_0.dtype

        tr = torch.rand(
            *timesteps_shape, 2, generator=torch_rng, device=device, dtype=dtype
        )
        tr = self.shift_timesteps(tr)

        t = tr.amax(dim=-1)
        r = tr.amin(dim=-1)

        x_1 = torch.randn_like(x_0)

        z = (1 - t) * x_0 + t * x_1

        # Target velocity v_g (Ground truth)
        v_g = x_1 - x_0

        # Predict Instantaneous Velocity (v) for JVP Tangent
        pred_inst, supplemental_outputs = net(z, t, t)
        v_inst = self.get_velocity(pred_inst, z, t)

        # JVP Calculation for Mean Flow
        # We need d(u_fn)/dt.
        # u_fn = (z - net(z, r, t))/t
        # Primals: (z, r, t)
        # Tangents: (dz/dt, dr/dt, dt/dt) -> (v_inst, 0, 1)

        def u_wrapper(z_in, r_in, t_in):
            out, _ = net(z_in, r_in, t_in)
            return self.get_velocity(out, z_in, t_in)

        # Ensure tangents are detached from graph (treated as constants for the differentiation)
        v_inst_detached = v_inst.detach()
        zeros_like_r = torch.zeros_like(r)
        ones_like_t = torch.ones_like(t)

        u, dudt = torch.func.jvp(
            u_wrapper, (z, r, t), (v_inst_detached, zeros_like_r, ones_like_t)
        )

        # Equation (7) of Pixel Meanflow
        # Construct Compound Vector Field V
        V = u + (t - r) * dudt.detach()

        # Compute Losses

        # Trajectory Loss (Mean Flow)
        loss_u_raw = F.mse_loss(V, v_g, reduction="none").mean(dim=(1, 2, 3))
        loss_u = self.adaptive_weighting(loss_u_raw).mean()

        # Instantaneous Loss (Auxiliary Flow Matching)
        # This is required to ensure v_inst (used in JVP) is accurate
        #
        # Note that this loss is not mentioned in the paper
        loss_v_raw = F.mse_loss(v_inst, v_g, reduction="none").mean(dim=(1, 2, 3))
        loss_v = self.adaptive_weighting(loss_v_raw).mean()

        total_loss = loss_u + loss_v

        # Equation (8) of Pixel Meanflow
        x_0_hat = z - u * t

        return (
            {
                "loss": total_loss,
                "loss_mf": loss_u,
                "loss_fm": loss_v,
            },
            {"x_0_hat": x_0_hat, "r": r, "t": t, **supplemental_outputs},
        )
