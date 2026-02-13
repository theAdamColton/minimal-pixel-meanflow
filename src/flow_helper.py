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
    # have instantaneous v-prediction
    instantaneous_velocity_proportion: float = 0.5

    uncondition_rate: float = 0.1
    max_cfg_scale: float = 6.0
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

    def convert_to_velocity(
        self, model_out: torch.Tensor, x_t: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        """
        Converts model output to velocity u based on prediction mode.
        """
        if self.conf.prediction_mode == "clean_input":
            u = (x_t - model_out) / t.clip(self.conf.eps)
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

            if self.conf.prediction_mode == "clean_input" and num_steps == 1:
                # skip velocity conversion and
                # simply return the direct single-step output
                return model_out

            u = self.convert_to_velocity(model_out, x_t, t_in)

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
        """
        4 forward passes w/o gradients,
        1 forward pass w/ grads
        """
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
        cfg_scale = self.conf.max_cfg_scale ** torch.rand(
            timesteps_shape, generator=torch_rng, device=device, dtype=dtype
        )

        # Disable CFG when the timestep is outside the cfg interval
        cfg_min_timestep, cfg_max_timestep = conf.cfg_interval
        cfg_scale = cfg_scale.masked_fill(
            (t < cfg_min_timestep) | (cfg_max_timestep < t), 1.0
        )

        x_1 = torch.randn_like(x_0)

        x_t = (1 - t) * x_0 + t * x_1

        # Target instantaneous velocity
        v_t = (x_t - x_0) / t.clip(self.conf.eps)

        with torch.no_grad():
            unconditional_labels = torch.full_like(labels, unc_label_id)
            pred, _ = net(
                torch.cat((x_t, x_t)),
                torch.cat((t, t)),
                torch.cat((t, t)),
                torch.cat((cfg_scale, cfg_scale)),
                torch.cat((labels, unconditional_labels)),
            )
            v_t_hat_cond_unc = self.convert_to_velocity(
                pred, torch.cat((x_t, x_t)), torch.cat((t, t))
            )
            v_t_hat_cond, v_t_hat_unc = v_t_hat_cond_unc.chunk(2)

            # Predicted instantaneous velocity using cfg
            v_t_hat_guided = v_t + (1 - 1 / cfg_scale) * (v_t_hat_cond - v_t_hat_unc)

            # Drop labels randomly
            uncondition_mask = (
                torch.rand(
                    (labels.shape[0],), generator=torch_rng, device=device, dtype=dtype
                )
                < conf.uncondition_rate
            )
            labels = labels.masked_fill(uncondition_mask, unc_label_id)
            # Predict unguided velocity when labels are absent
            v_target = masked_fill(
                unsqueeze_trailing(uncondition_mask, v_t), v_t, v_t_hat_guided
            )

            # JVP Calculation for Mean Flow
            # We need d(u_fn)/dt.
            # u_fn = (z - net(z, r, t))/t
            # Primals: (z, r, t)
            # Tangents: (dz/dt, dr/dt, dt/dt) -> (v_inst, 0, 1)

            def u_wrapper(x_in, r_in, t_in):
                out, _ = net(x_in, r_in, t_in, cfg_scale, labels)
                return self.convert_to_velocity(out, x_in, t_in)

            # TODO, The paper and the official iMF compute the trajectory u,
            # and dudt in a single call to jvp. However, jvp doesn't
            # play nice with torch's autocast and backwards differentiation,
            # so as a temporary fix I compute dudt and u seperately

            # TODO there is a discrepancy between pixel-mean-flow's
            # alogorithm 2, and improved-mean-flow's code. I follow the code's
            # implementation and use the conditioned velocity as the jvp tangents
            #
            # Another difference is that iMF's code evaluates the tanget using
            # the labels and the cfg scale, but the algorithm indicates that
            # the tangent is evaluated with `0.0` for CFG and null label.
            # I follow the iMF code and use the cfg and label.

            zeros_like_r = torch.zeros_like(r)
            ones_like_t = torch.ones_like(t)
            _, dudt = torch.func.jvp(
                u_wrapper, (x_t, r, t), (v_t_hat_cond, zeros_like_r, ones_like_t)
            )

        prediction, supplemental_outputs = net(x_t, r, t, cfg_scale, labels)
        u = self.convert_to_velocity(prediction, x_t, t)

        # Equation (7) of Pixel Meanflow
        # Construct Compound Vector Field V
        V = u + (t - r) * dudt

        # Compute Losses

        # Trajectory Loss (improved mean flow)
        loss_u_raw = F.mse_loss(V, v_target, reduction="none").mean(dim=(1, 2, 3))
        loss_u = self.adaptive_weighting(loss_u_raw).mean()

        total_loss = loss_u

        # Equation (8) of Pixel Meanflow
        # (Assuming r=0)
        x_0_hat = x_t - u * t

        return (
            {
                "loss": total_loss,
                "loss_mf": loss_u,
            },
            {"x_0_hat": x_0_hat, "r": r, "t": t, **supplemental_outputs},
        )
