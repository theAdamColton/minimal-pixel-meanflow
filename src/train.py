from dataclasses import dataclass
from functools import partial
from einops import rearrange, repeat
import torch
import datasets
from torchvision import transforms
from src.conf import MainConfig
from src.flow_helper import FlowHelper
from src.net import ViTDenoiser
from src.supplemental_net import ConvNextV2Loss, LPIPSLoss


def _classify_params(model: ViTDenoiser):
    nonhidden_weights = []
    biases = []
    hidden_weights = []

    for name, parameter in model.named_parameters():
        is_hidden = "blocks." in name
        is_multidim = parameter.ndim == 2

        if not is_multidim:
            biases.append(parameter)
        elif is_hidden:
            hidden_weights.append(parameter)
        else:
            nonhidden_weights.append(parameter)

    adamw_param_groups = [
        {"params": nonhidden_weights, "use_weight_decay": True},
        {"params": biases, "use_weight_decay": False},
    ]
    muon_param_groups = [
        {"params": hidden_weights, "use_weight_decay": True},
    ]

    return adamw_param_groups, muon_param_groups


def pil_to_tensor(x):
    x = x.convert("RGB")
    w, h = x.size
    x = x.tobytes()
    x = torch.frombuffer(x, dtype=torch.uint8).reshape(h, w, 3)
    return x


def forward_model(
    model: ViTDenoiser,
    patch_size: int,
    x_t: torch.Tensor,
    r: torch.Tensor,
    t: torch.Tensor,
) -> torch.Tensor:
    b, h, w, c = x_t.shape
    device, dtype = x_t.device, x_t.dtype
    nph = h // patch_size
    npw = w // patch_size

    h_coords = torch.linspace(-1.0, 1.0, nph, device=device)
    w_coords = torch.linspace(-1.0, 1.0, npw, device=device)
    patch_coords = torch.stack(torch.meshgrid(h_coords, w_coords, indexing="ij"), -1)
    patch_coords = repeat(patch_coords, "nph npw nd -> b (nph npw) nd", b=1)

    patches = rearrange(
        x_t,
        "b (nph ph) (npw pw) c -> b (nph npw) (ph pw c)",
        ph=patch_size,
        pw=patch_size,
    )

    r = r.squeeze()[:, None]
    t = t.squeeze()[:, None]

    output = model(
        patches=patches, terminal_timesteps=r, timesteps=t, patch_coords=patch_coords
    )

    x_0_hat = output.prediction

    x_0_hat = rearrange(
        x_0_hat,
        "b (nph npw) (ph pw c) -> b (nph ph) (npw pw) c",
        nph=nph,
        npw=npw,
        c=3,
        ph=patch_size,
        pw=patch_size,
    )

    return x_0_hat


def train(conf: MainConfig):
    dataset = datasets.load_dataset(conf.dataset_path_or_url)

    def apply_transforms(examples):
        images = examples.pop("image")
        examples["pixel_values"] = [pil_to_tensor(image) for image in images]
        return examples

    train_dataset = dataset["train"].with_transform(apply_transforms)
    test_dataset = dataset["validation"].with_transform(apply_transforms)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=conf.batch_size,
        shuffle=True,
        num_workers=conf.num_workers,
        drop_last=True,
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=conf.batch_size,
        shuffle=False,
        num_workers=conf.num_workers,
    )

    model = ViTDenoiser(conf.model)
    ema_model = ViTDenoiser(conf.model)
    ema_model.load_state_dict(model.state_dict())

    lpips_loss_fn = LPIPSLoss().to(conf.device, conf.dtype)
    convnext_loss_fn = ConvNextV2Loss().to(conf.device, conf.dtype)

    flow_helper = FlowHelper(conf.flow)

    model = model.to(conf.device)
    ema_model = ema_model.to(conf.device).requires_grad_(False)

    adamw_params, muon_params = _classify_params(model)

    optim_muon = torch.optim.Muon(muon_params)
    optim_adamw = torch.optim.AdamW(adamw_params)

    global_step = 0

    def set_optim_hparams_():
        warmup_p = min(global_step / conf.num_warmup_steps, 1.0)

        for p in optim_muon.param_groups:
            p["lr"] = conf.lr_muon * warmup_p
            p["weight_decay"] = conf.weight_decay_muon
            p["momentum"] = conf.momentum_muon

        for p in optim_adamw.param_groups:
            p["lr"] = conf.lr_adamw * warmup_p
            p["weight_decay"] = (
                conf.weight_decay_adamw if p["use_weight_decay"] else 0.0
            )
            p["betas"] = conf.betas_adamw

    for epoch in range(conf.num_train_epochs):
        for batch in train_loader:

            def _step():
                pixel_values = batch.pop("pixel_values")
                pixel_values = (
                    pixel_values.to(conf.device, conf.dtype).div_(255).mul_(2).sub_(1)
                )

                with torch.autocast(conf.device.type, conf.dtype):
                    meanflow_loss_dict = flow_helper.compute_meanflow_loss(
                        x_0=pixel_values,
                        net=partial(forward_model, model, conf.patch_size),
                        timesteps_shape=(pixel_values.shape[0], 1, 1, 1),
                    )

                    x_0_hat = meanflow_loss_dict["x_0_hat"]
                    x_0_hat = rearrange(x_0_hat, "b h w c -> b c h w")
                    pixel_values = rearrange(pixel_values, "b h w c -> b c h w")

                    lpips_loss = lpips_loss_fn(x_0_hat, pixel_values)
                    convnext_loss = convnext_loss_fn(x_0_hat, pixel_values)

                    total_loss = (
                        meanflow_loss_dict["loss"].float()
                        + lpips_loss.float() * conf.lpips_weight
                        + convnext_loss.float() * conf.convnext_weight
                    )

                total_loss.backward()
                set_optim_hparams_()
                optim_adamw.step()
                optim_adamw.zero_grad(set_to_none=True)
                optim_muon.step()
                optim_muon.zero_grad(set_to_none=True)

                log_dict = {
                    "loss_meanflow": meanflow_loss_dict["loss"],
                    "loss_lpips": lpips_loss,
                    "loss_convnext": convnext_loss,
                    "loss_total": total_loss,
                }
                log_dict = {k: v.detach().cpu().item() for k, v in log_dict.items()}

                return log_dict

            log_dict = _step()

            print(log_dict)
