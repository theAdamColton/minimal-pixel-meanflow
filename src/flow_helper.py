from dataclasses import dataclass
import math
from typing import Callable, Literal, Tuple, Dict

import torch
import torch.nn.functional as F


def unsqueeze_trailing(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    while x.ndim < y.ndim:
        x = x.unsqueeze(-1)
    return x


def masked_fill(mask, x, y):
    return x * mask + y * ~mask


@dataclass
class FlowHelperConfig:
    time_shift_dim: float = 12288
    time_shift_base: float = 4096
    eps: float = 1e-3
    prediction_mode: Literal["clean_input", "velocity"] = "clean_input"

    # This proportion of each batch will
    # have instantaneous v-pred loss
    instantaneous_velocity_proportion: float = 0.5

    uncondition_rate: float = 0.1
    max_cfg_scale: float = 7.0
    # Enable CFG guidance only when timesteps are in this range
    cfg_interval: tuple[float, float] = (0.1, 0.7)

    norm_p: float = 1.0
    norm_eps: float = 0.01


class FlowHelper:
    """
    Flow from noise (t=1.0) to data (t=0.0).
    """

    def __init__(self, conf: FlowHelperConfig = FlowHelperConfig()):
        self.conf = conf

    def shift_timesteps(self, t: torch.Tensor):
        """
        Shifts timesteps towards noise (1.0) based on resolution/dimension.
        """
        shift = math.sqrt(self.conf.time_shift_dim / self.conf.time_shift_base)
        t = 1 - (1 - t) / (1 - t + shift * t)
        return t

    def get_velocity(
        self, model_out: torch.Tensor, x_t: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        """
        Converts model output to velocity u based on prediction mode.
        """
        if self.conf.prediction_mode == "clean_input":
            t_clipped = t.clamp(min=self.conf.eps)
            u = (x_t - model_out) / t_clipped
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
        # Net signature: (x_t, r, t, cfg, labels) -> (Prediction (x_0 or v), dict)
        net: Callable[
            [torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
            tuple[torch.Tensor, dict],
        ],
        x_1: torch.Tensor,
        labels: torch.Tensor,
        cfg: torch.Tensor,
        num_steps: int = 4,
    ):
        device = x_1.device
        dtype = x_1.dtype

        t = torch.linspace(1.0, 0.0, num_steps + 1, device=device, dtype=dtype)

        x_t = x_1

        for i in range(num_steps):
            t_curr = t[i]
            t_next = t[i + 1]
            dt = t_next - t_curr

            t_in = unsqueeze_trailing(t_curr, x_t)
            r_in = unsqueeze_trailing(t_next, x_t)

            model_out, _ = net(x_t, r_in, t_in, cfg, labels)
            u = self.get_velocity(model_out, x_t, t_in)

            x_t = x_t + u * dt

        return x_t

    def compute_meanflow_loss(
        self,
        x_0: torch.Tensor,
        labels: torch.Tensor,
        # Net signature: (x_t, r, t, cfg, labels) -> (Prediction (x_0 or v), dict)
        net: Callable[
            [torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
            tuple[torch.Tensor, dict],
        ],
        timesteps_shape: tuple,
        torch_rng: torch.Generator | None = None,
        unc_label_id: int = 1023,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        device, dtype = x_0.device, x_0.dtype
        conf = self.conf

        # Sample timestep and terminal timestep
        tr = torch.rand(
            *timesteps_shape, 2, generator=torch_rng, device=device, dtype=dtype
        )
        tr = self.shift_timesteps(tr)

        t = tr.amax(dim=-1)
        r = tr.amin(dim=-1)

        # Sample flow matching mask:
        # Model will have to predict instantaneous velocity for these samples
        flow_matching_mask = (
            torch.rand(timesteps_shape, generator=torch_rng, device=device, dtype=dtype)
            < conf.instantaneous_velocity_proportion
        )
        r = masked_fill(flow_matching_mask, t, r)

        # Sample CFG scale
        cfg_scale = torch.rand(
            timesteps_shape, generator=torch_rng, device=device, dtype=dtype
        ).pow(self.conf.max_cfg_scale)

        # Disable CFG when the timestep is outside the cfg interval
        cfg_min_timestep, cfg_max_timestep = conf.cfg_interval
        cfg_scale = cfg_scale.masked_fill(
            (t < cfg_min_timestep) | (cfg_max_timestep < t), 1.0
        )

        unconditional_labels = torch.full_like(labels, unc_label_id)

        x_1 = torch.randn_like(x_0)

        x_t = (1 - t) * x_0 + t * x_1

        # Target instantaneous velocity
        v = x_1 - x_0

        # Predicted instantaneous velocity using cfg,
        # and save supplemental_outputs when forwarding with labels
        pred_inst_cond, supplemental_outputs = net(x_t, t, t, cfg_scale, labels)
        v_hat_cond = self.get_velocity(pred_inst_cond, x_t, t)
        pred_inst_unc, _ = net(x_t, t, t, cfg_scale, unconditional_labels)
        v_hat_unc = self.get_velocity(pred_inst_unc, x_t, t)
        v_hat_guided = v + (1 - 1 / cfg_scale) * (v_hat_cond - v_hat_unc)

        # Drop labels randomly
        uncondition_mask = (
            torch.rand(
                (labels.shape[0],), generator=torch_rng, device=device, dtype=dtype
            )
            < conf.uncondition_rate
        )
        labels = labels.masked_fill(uncondition_mask, unc_label_id)
        # Predict unguided velocity/trajectory when labels are absent
        v_target = masked_fill(unsqueeze_trailing(uncondition_mask, v), v, v_hat_guided)

        # JVP Calculation for Mean Flow
        # We need d(u_fn)/dt.
        # u_fn = (z - net(z, r, t))/t
        # Primals: (z, r, t)
        # Tangents: (dz/dt, dr/dt, dt/dt) -> (v_inst, 0, 1)

        def u_wrapper(x_in, r_in, t_in):
            out, _ = net(x_in, r_in, t_in, cfg_scale, labels)
            return self.get_velocity(out, x_in, t_in)

        # TODO there is a discrepancy between pixel-mean-flow's
        # alogorithm 2, and improved-mean-flow's code. I follow the code's
        # implementation and use the conditioned velocity as the jvp tangents

        zeros_like_r = torch.zeros_like(r)
        ones_like_t = torch.ones_like(t)
        u, dudt = torch.func.jvp(
            u_wrapper, (x_t, r, t), (v_hat_cond, zeros_like_r, ones_like_t)
        )

        # Equation (7) of Pixel Meanflow
        # Construct Compound Vector Field V
        V = u + (t - r) * dudt.detach()

        # Compute Losses

        # Trajectory Loss (improved mean flow)
        loss_u_raw = F.mse_loss(V, v_target.detach(), reduction="none").mean(
            dim=(1, 2, 3)
        )
        loss_u = self.adaptive_weighting(loss_u_raw).mean()

        # # Instantaneous Loss (Auxiliary Flow Matching)
        # # This is required to ensure v_inst (used in JVP) is accurate
        # #
        # # Note that this loss is not mentioned in the paper but is
        # # in the official Mean-flow and Improved-Mean-flow code
        # loss_v_raw = F.mse_loss(v_hat_guided, v_target.detach(), reduction="none").mean(
        #     dim=(1, 2, 3)
        # )
        # loss_v = self.adaptive_weighting(loss_v_raw).mean()

        total_loss = loss_u  # + loss_v

        # Equation (8) of Pixel Meanflow
        x_0_hat = x_t - u * t

        return (
            {
                "loss": total_loss,
                "loss_mf": loss_u,
                # "loss_fm": loss_v,
            },
            {"x_0_hat": x_0_hat, "r": r, "t": t, **supplemental_outputs},
        )
