import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange
import transformers
import lpips


class DinoV3Encoder(nn.Module):
    def __init__(
        self, dino_path_or_url: str = "facebook/dinov3-vitb16-pretrain-lvd1689m"
    ):
        super().__init__()
        self.dino = (
            transformers.DINOv3ViTModel.from_pretrained(dino_path_or_url)
            .eval()
            .requires_grad_(False)
        )
        self.hidden_size = self.dino.config.hidden_size
        self.patch_size = self.dino.config.patch_size

        # Imagenet pixelwise mean and std
        self.register_buffer(
            "mean",
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "std",
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1),
            persistent=False,
        )

    def forward(self, pixel_values: torch.Tensor):
        # Expects [..., h, w, c] in [-1, 1]
        # Returns flattened patch features, [..., S, D]
        *leading, h, w, c = pixel_values.shape
        pixel_values = rearrange(pixel_values, "... h w c -> (...) c h w")

        pixel_values = (pixel_values + 1) / 2
        pixel_values = (pixel_values - self.mean) / self.std

        features = self.dino(pixel_values).last_hidden_state

        # skip register tokens
        # Assuming format [CLS, PATCHES...]
        cls_feature = features[:, 0]
        patch_features = features[:, 5:]  # skip sidechannel features

        # discard affine weights by renormalizing
        patch_features = F.layer_norm(patch_features, (patch_features.shape[-1],))
        cls_feature = F.layer_norm(cls_feature, (cls_feature.shape[-1],))

        patch_features = patch_features.reshape(
            *leading, patch_features.shape[1], patch_features.shape[2]
        )
        cls_feature = cls_feature.reshape(*leading, cls_feature.shape[-1])

        return cls_feature, patch_features


class BatchedRandomResizedCrop(torch.nn.Module):
    def __init__(
        self,
        size,
        scale=(0.08, 1.0),
        ratio=(3.0 / 4.0, 4.0 / 3.0),
        resize_mode: str = "nearest",
        torch_rng: torch.Generator | None = None,
    ):
        super().__init__()
        self.size = size if isinstance(size, tuple) else (size, size)
        self.scale = scale
        self.ratio = ratio
        self.resize_mode = resize_mode
        self.torch_rng = torch_rng

    def get_params(self, batch_size, height, width):
        area = height * width

        # Randomly sample scales and ratios for the whole batch
        log_ratio = torch.log(torch.tensor(self.ratio))

        # Generate random scales and ratios
        scales = torch.empty(batch_size).uniform_(*self.scale, generator=self.torch_rng)
        ratios = torch.exp(
            torch.empty(batch_size).uniform_(*log_ratio, generator=self.torch_rng)
        )

        target_areas = scales * area
        w = torch.round(torch.sqrt(target_areas * ratios)).long()
        h = torch.round(torch.sqrt(target_areas / ratios)).long()

        # Ensure crops aren't larger than the original image
        w = torch.clamp(w, max=width)
        h = torch.clamp(h, max=height)

        # Randomly sample top-left corners
        i = torch.empty(batch_size).uniform_(0, 1, generator=self.torch_rng) * (
            height - h
        )
        j = torch.empty(batch_size).uniform_(0, 1, generator=self.torch_rng) * (
            width - w
        )

        return i.long(), j.long(), h, w

    def forward(self, img_batch: torch.Tensor, params=None):
        B, C, H, W = img_batch.shape

        device, dtype = img_batch.device, img_batch.dtype

        if params is None:
            i, j, h, w = self.get_params(B, H, W)
        else:
            i, j, h, w = params

        i = i.to(device)
        j = j.to(device)
        h = h.to(device)
        w = w.to(device)

        # Normalize coordinates to [-1, 1] for grid_sample
        # We create a grid for the target size and map it back to the crop regions
        grid_h, grid_w = self.size

        # Generate base normalized meshgrid
        rows = torch.linspace(-1, 1, grid_h, device=device, dtype=dtype)
        cols = torch.linspace(-1, 1, grid_w, device=device, dtype=dtype)
        mesh_y, mesh_x = torch.meshgrid(
            rows, cols, indexing="ij"
        )  # [target_H, target_W]

        # Flatten and expand to batch size
        mesh_x = mesh_x.reshape(1, -1).expand(B, -1)
        mesh_y = mesh_y.reshape(1, -1).expand(B, -1)

        # Map target [-1, 1] to the specific crop window in original image pixels
        # x_orig = j + (mesh_x + 1) * w / 2
        # Then convert x_orig back to [-1, 1] relative to the WHOLE original image
        scale_x = w.unsqueeze(1) / W
        offset_x = (2 * j.unsqueeze(1) + w.unsqueeze(1)) / W - 1

        scale_y = h.unsqueeze(1) / H
        offset_y = (2 * i.unsqueeze(1) + h.unsqueeze(1)) / H - 1

        final_grid_x = mesh_x * scale_x + offset_x
        final_grid_y = mesh_y * scale_y + offset_y

        # Reshape to [B, target_H, target_W, 2]
        grid = torch.stack([final_grid_x, final_grid_y], dim=-1).reshape(
            B, grid_h, grid_w, 2
        )

        grid = grid.to(dtype)

        return F.grid_sample(
            img_batch, grid, mode=self.resize_mode, align_corners=False
        )


class ConvNextV2Loss(nn.Module):
    def __init__(
        self,
        path_or_url: str = "facebook/convnextv2-base-22k-224",
        feature_indices: list[int] | None = None,
        crop_scale: tuple[float, float] = (0.3, 1.0),
        crop_size: tuple[int, int] | None = None,
    ):
        super().__init__()

        self.preprocessor = transformers.AutoImageProcessor.from_pretrained(path_or_url)

        # Imagenet pixelwise mean and std
        self.register_buffer(
            "mean",
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "std",
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1),
            persistent=False,
        )

        if crop_size is not None:
            self.crop_size = crop_size
        else:
            self.crop_size = self.preprocessor.size["shortest_edge"]
        self.random_resized_crop = BatchedRandomResizedCrop(
            self.crop_size, scale=crop_scale
        )

        classification_model = (
            transformers.ConvNextV2ForImageClassification.from_pretrained(path_or_url)
        )
        self.model = classification_model.convnextv2
        self.model.requires_grad_(False).eval()

        if feature_indices is None:
            feature_indices = list(range(self.model.config.num_stages))
        self.feature_indices = feature_indices

    def forward(
        self, pixel_values_synth: torch.Tensor, pixel_values_real: torch.Tensor
    ):
        # Expects [B, C, H, W] in (-1,1)
        b, _, h, w = pixel_values_real.shape

        crop_params = self.random_resized_crop.get_params(b, h, w)

        pixel_values_synth = self.random_resized_crop(pixel_values_synth, crop_params)
        pixel_values_real = self.random_resized_crop(pixel_values_real, crop_params)

        pixel_values = torch.cat((pixel_values_synth, pixel_values_real))

        pixel_values = (pixel_values + 1) / 2
        pixel_values = (pixel_values - self.mean) / self.std

        loss = 0.0

        hidden_states = self.model.embeddings(pixel_values)
        for i, layer_module in enumerate(self.model.encoder.stages):
            hidden_states = layer_module(hidden_states)

            if i in self.feature_indices:
                hidden_states_synth, hidden_states_real = hidden_states.chunk(2, dim=0)
                layer_loss = -F.cosine_similarity(
                    hidden_states_synth, hidden_states_real
                ).mean()
                loss += layer_loss / len(self.feature_indices)

        return loss


class LPIPSLoss(nn.Module):
    def __init__(
        self, crop_scale: tuple[float, float] = (0.3, 1.0), crop_size: int = 224
    ):
        super().__init__()
        self.random_resized_crop = BatchedRandomResizedCrop(crop_size, scale=crop_scale)
        self.lpips = lpips.LPIPS(net="vgg").requires_grad_(False).eval()

    def forward(
        self, pixel_values_synth: torch.Tensor, pixel_values_real: torch.Tensor
    ):
        # Expects [B, C, H, W] in (-1,1)
        b, _, h, w = pixel_values_real.shape

        crop_params = self.random_resized_crop.get_params(b, h, w)

        pixel_values_synth = self.random_resized_crop(pixel_values_synth, crop_params)
        pixel_values_real = self.random_resized_crop(pixel_values_real, crop_params)

        loss = self.lpips(pixel_values_synth, pixel_values_real).mean()

        return loss


class REPAProjector2D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, 1, 1)

    def forward(self, x):
        # Expects [B, D, H, W]
        return self.conv(x)
