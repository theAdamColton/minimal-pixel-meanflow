import math
import json
from pathlib import Path
from functools import partial

from einops import rearrange, repeat
from tqdm import tqdm
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import datasets
import torchvision

from src.conf import MainConfig
from src.flow_helper import FlowHelper
from src.net import ViTDenoiser
from src.supplemental_net import ConvNextV2Loss, LPIPSLoss


def _lerp(a, b, p):
    return (b - a) * p + a


def _find_closest_factors(b):
    # Start searching from the floor of the square root
    start = int(math.sqrt(b))

    for nh in range(start, 0, -1):
        if b % nh == 0:
            nw = b // nh
            return nh, nw
    raise ValueError()


def _classify_params(model: ViTDenoiser):
    nonhidden_weights = []
    biases = []
    hidden_weights = []

    for name, parameter in model.named_parameters():
        is_hidden = "blocks." in name
        is_2d = parameter.ndim == 2

        if not is_2d:
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

    r = r.squeeze().unsqueeze(-1)
    t = t.squeeze().unsqueeze(-1)

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

    lpips_loss_fn = None
    if conf.lpips_weight > 0:
        lpips_loss_fn = LPIPSLoss().to(conf.device, conf.dtype)
    convnext_loss_fn = None
    if conf.convnext_weight > 0:
        convnext_loss_fn = ConvNextV2Loss().to(conf.device, conf.dtype)

    flow_helper = FlowHelper(conf.flow)

    model = model.to(conf.device)
    ema_model = ema_model.to(conf.device).requires_grad_(False)

    adamw_p_groups, muon_p_groups = _classify_params(model)

    # optim_muon = torch.optim.Muon(muon_p_groups, adjust_lr_fn="match_rms_adamw")
    optim_adamw = torch.optim.AdamW(adamw_p_groups + muon_p_groups)

    global_step = 0

    output_path = Path("out/")
    output_path.mkdir(exist_ok=True)
    run_num = len(list(output_path.iterdir()))
    run_path = output_path / f"{run_num:05}"
    run_path.mkdir()

    input_shape = None

    def _step(batch):
        pixel_values = batch.pop("pixel_values")
        pixel_values = (
            pixel_values.to(conf.device, conf.dtype).div_(255).mul_(2).sub_(1)
        )
        nonlocal input_shape
        input_shape = pixel_values.shape

        with torch.autocast(
            conf.device.type, conf.dtype, enabled=conf.dtype != torch.float32
        ):
            loss_dict, extra_dict = flow_helper.compute_meanflow_loss(
                x_0=pixel_values,
                net=partial(forward_model, model, conf.patch_size),
                timesteps_shape=(pixel_values.shape[0], 1, 1, 1),
            )

            # x_0_hat = extra_dict["x_0_hat"]
            # x_0_hat = rearrange(x_0_hat, "b h w c -> b c h w")
            # pixel_values = rearrange(pixel_values, "b h w c -> b c h w")

            lpips_loss = 0.0
            if conf.lpips_weight > 0.0:
                lpips_loss = lpips_loss_fn(x_0_hat, pixel_values)
            convnext_loss = 0.0
            if conf.convnext_weight > 0.0:
                convnext_loss = convnext_loss_fn(x_0_hat, pixel_values)

            total_loss = (
                loss_dict["loss"]
                + lpips_loss * conf.lpips_weight
                + convnext_loss * conf.convnext_weight
            )

        total_loss.backward()

        for p_group in adamw_p_groups, muon_p_groups:
            for parameters in p_group:
                torch.nn.utils.clip_grad_norm_(parameters["params"], max_norm=1.0)

        warmup_p = _lerp(0.1, 1.0, min(global_step / conf.num_warmup_steps, 1.0))

        # for p in optim_muon.param_groups:
        #     p["lr"] = conf.lr_muon * warmup_p
        #     p["weight_decay"] = conf.weight_decay_muon
        #     p["momentum"] = conf.momentum_muon

        lr_adamw = conf.lr_adamw * warmup_p
        for p in optim_adamw.param_groups:
            p["lr"] = lr_adamw
            p["weight_decay"] = (
                conf.weight_decay_adamw if p["use_weight_decay"] else 0.0
            )
            p["betas"] = conf.betas_adamw

        optim_adamw.step()
        # optim_muon.step()
        optim_adamw.zero_grad(set_to_none=True)
        # optim_muon.zero_grad(set_to_none=True)

        ema_beta = warmup_p * conf.ema_beta
        with torch.no_grad():
            for ema_p, p in zip(ema_model.parameters(), model.parameters()):
                ema_p.lerp_(p, 1 - ema_beta)

        log_dict = {
            **loss_dict,
            "loss_lpips": lpips_loss,
            "loss_convnext": convnext_loss,
            "loss_total": total_loss,
            "lr_adamw": lr_adamw,
        }

        return log_dict

    if conf.should_compile:
        _step = torch.compile(_step)

    for epoch in range(conf.num_train_epochs):
        for batch in tqdm(train_loader):
            log_dict = _step(batch)
            log_dict = {
                k: v.detach().cpu().item() if isinstance(v, torch.Tensor) else v
                for k, v in log_dict.items()
            }

            log_dict["epoch"] = epoch
            log_dict["global_step"] = global_step

            @torch.inference_mode()
            def _validate():
                torch_rng = torch.Generator(conf.device)
                x_1 = torch.randn(*input_shape, device=conf.device, dtype=conf.dtype)
                with torch.autocast(conf.device.type, conf.dtype):
                    x_0 = flow_helper.sample_euler(
                        partial(forward_model, ema_model, conf.patch_size),
                        x_1=x_1,
                        num_steps=1,
                    )

                x_0 = (
                    x_0.float()
                    .add(1)
                    .div(2)
                    .clip(0, 1)
                    .mul(255)
                    .round()
                    .to(torch.uint8)
                    .cpu()
                )

                pad_amount = 2
                h_padding = torch.zeros(
                    x_0.shape[0],
                    pad_amount,
                    x_0.shape[2],
                    x_0.shape[3],
                    dtype=torch.uint8,
                )
                x_0 = torch.cat((x_0, h_padding), 1)
                w_padding = torch.zeros(
                    x_0.shape[0],
                    x_0.shape[1],
                    pad_amount,
                    x_0.shape[3],
                    dtype=torch.uint8,
                )
                x_0 = torch.cat((x_0, w_padding), 2)

                b = input_shape[0]
                nh, nw = _find_closest_factors(b)
                x_0 = rearrange(x_0, "(nh nw) h w c -> c (nh h) (nw w)", nh=nh, nw=nw)
                x_0 = x_0[:, :-pad_amount, :-pad_amount]
                torchvision.io.write_png(x_0, str(run_path / f"{global_step:06}.png"))

            should_validate = global_step % conf.validate_every_num_steps == 0
            if should_validate:
                _validate()

            with open(run_path / "log.jsonl", "a") as f:
                f.write(json.dumps(log_dict) + "\n")

            if should_validate:
                d = pd.read_json(run_path / "log.jsonl", lines=True)
                d["loss_total_smoothed"] = (
                    d["loss_total"].ewm(adjust=False, alpha=0.005).mean()
                )
                if len(d) > 1:
                    plt.plot(d["global_step"], d["loss_total_smoothed"])
                    plt.plot(d["global_step"], d["loss_total"], alpha=0.1)
                    plt.ylim(
                        d["loss_total"].min(),
                        d["loss_total_smoothed"].quantile(0.9),
                    )
                    plt.savefig(run_path / "total_loss.png")
                    plt.close()

            global_step += 1
