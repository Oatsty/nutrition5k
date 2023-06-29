from abc import ABC, abstractmethod
from einops import rearrange
import numpy as np
import timm
import torch
from torch import nn
import torch.nn.functional as f
from transformers import ResNetModel, SwinModel, AutoFeatureExtractor, ViTFeatureExtractor  # type: ignore
from torchvision.ops import FeaturePyramidNetwork

from yacs.config import CfgNode as CN

from seg_openseed import OpenSeeDSeg


class BaseRegressor(nn.Module, ABC):
    @abstractmethod
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def __call__(self, x: torch.Tensor, **kwargs) -> dict[str, torch.Tensor]:
        return self.forward(x, **kwargs)


class Regressor(BaseRegressor):
    def __init__(self, in_dim: int, hidden_dim: int, dropout_rate: float = 0.1) -> None:
        super(Regressor, self).__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
        )
        self.regress1 = nn.ModuleDict(
            {
                x: nn.Sequential(
                    nn.Linear(hidden_dim, 1),
                )
                for x in ["cal", "mass", "fat", "carb", "protein"]
            }
        )
        self.regress2 = nn.ModuleDict(
            {
                x: nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    # nn.Dropout(dropout_rate),
                    nn.Linear(hidden_dim, 1),
                )
                for x in ["cal", "mass", "fat", "carb", "protein"]
            }
        )

    def forward(self, x: torch.Tensor, depth: int = 3):
        x = self.fc1(x)
        if depth == 2:
            out = {
                d: self.regress1[d](x)
                for d in ["cal", "mass", "fat", "carb", "protein"]
            }
        elif depth == 3:
            out = {
                d: self.regress2[d](x)
                for d in ["cal", "mass", "fat", "carb", "protein"]
            }
        else:
            raise ValueError(f"Invalid depth: {depth}")
        return out


class RegressorIngrs(BaseRegressor):
    def __init__(self, in_dim, hidden_dim) -> None:
        super(RegressorIngrs, self).__init__()
        self.regress_ingrs = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 555),
        )
        self.fc1 = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
        )
        self.regress = nn.ModuleDict(
            {
                x: nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, 1),
                )
                for x in ["cal", "mass", "fat", "carb", "protein"]
            }
        )

    def forward(self, x):
        out_ingrs = self.regress_ingrs(x)
        x = self.fc1(x)
        out = {d: self.regress[d](x) for d in ["cal", "mass", "fat", "carb", "protein"]}
        out["ingrs"] = out_ingrs
        return out


class SelfAttention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head**-0.5
        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim=-1)

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_k = nn.Linear(dim, inner_dim, bias=False)
        self.to_v = nn.Linear(dim, inner_dim, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
        self.to_v.weight.data *= 0.67
        self.to_out.weight.data *= 0.67

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q = self.to_q(x)
        k = self.to_k(x)
        v = self.to_v(x)
        q, k, v = map(
            lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.heads), [q, k, v]
        )

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)

        out = torch.matmul(attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.to_out(out)
        out = out + x
        # out = self.norm(out + x)
        return out


class BaseModel(nn.Module, ABC):
    @abstractmethod
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def __call__(
        self, rgb_img: torch.Tensor, depth_img: torch.Tensor, **kwargs
    ) -> dict[str, torch.Tensor]:
        return self.forward(rgb_img, depth_img, **kwargs)


class SimpleInceptionV2(BaseModel):
    def __init__(
        self,
        hidden_dim,
        pretrained_model: str = "inception_resnet_v2",
        regressor: nn.Module = Regressor(6144, 4096),
    ) -> None:
        super(SimpleInceptionV2, self).__init__()
        self.conv0 = nn.Conv2d(4, 3, 1)
        self.backbone = timm.create_model(
            pretrained_model, features_only=True, pretrained=True
        )
        self.pooling = nn.AvgPool2d(3, 2)
        self.flatten = nn.Flatten()
        self.regressor = regressor

    def forward(self, img: torch.Tensor, depth: torch.Tensor, **kwargs):
        x = torch.cat([img, depth], dim=1)
        x = self.conv0(x)
        x = self.backbone(x)[-1]
        x = self.pooling(x)
        x = self.flatten(x)
        out = self.regressor(x)
        return out


class FPN(BaseModel):
    def __init__(
        self,
        hidden_dim: int,
        pretrained_model: str = "microsoft/resnet-50",
        resolution_level: int = 2,
        regressor: BaseRegressor = Regressor(2048, 2048),
    ) -> None:
        super(FPN, self).__init__()
        self.backbone = nn.ModuleDict({x: ResNetModel.from_pretrained(pretrained_model) for x in ["rgb_img", "depth_img"]})  # type: ignore
        channel_list = [256, 512, 1024, 2048]
        self.resolution_level = resolution_level
        self.fpn = FeaturePyramidNetwork(channel_list, hidden_dim)
        self.attention = SelfAttention(hidden_dim)
        self.pooling = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()
        self.regressor = regressor
        for module in self.regressor.modules():
            if isinstance(module, nn.Linear):
                module.weight.data *= 0.67

    def forward(
        self, rgb_img: torch.Tensor, depth_img: torch.Tensor, skip_attn: bool = False
    ):
        B, _, _, _ = rgb_img.shape
        depth_img = depth_img.expand(-1, 3, -1, -1)
        inputs = {"rgb_img": rgb_img, "depth_img": depth_img}
        hidden_states_list = []
        for key, x in inputs.items():
            assert isinstance(self.backbone[key], ResNetModel)
            hidden_states = self.backbone[key](
                x, output_hidden_states=True
            ).hidden_states
            hidden_states = hidden_states[1:]
            hidden_states_list.append(hidden_states)
        hidden_states_dict = {
            i: hid1 + hid2 for i, (hid1, hid2) in enumerate(zip(*hidden_states_list))
        }
        hidden_states = self.fpn(hidden_states_dict)
        fin_res = hidden_states[self.resolution_level].shape[-1]
        for i, hid in hidden_states.items():
            if i <= self.resolution_level:
                hidden_states[i] = f.adaptive_avg_pool2d(hid, fin_res)
            else:
                hidden_states[i] = f.interpolate(hid, fin_res)
        hidden_states = torch.stack([emb for emb in hidden_states.values()])
        if not skip_attn:
            pooled_hidden = hidden_states.mean(0)
            pooled_hidden = rearrange(pooled_hidden, "b c h w -> b (h w) c", h=fin_res)
            pooled_hidden = self.attention(pooled_hidden)
            pooled_hidden = rearrange(pooled_hidden, "b (h w) c -> b c h w", h=fin_res)
            hidden_states = hidden_states + pooled_hidden.unsqueeze(0)
        hidden_states = rearrange(hidden_states, "n b c h w -> b (n c) h w", b=B)
        emb = self.pooling(hidden_states)
        emb = self.flatten(emb)
        out = self.regressor(emb)
        return out


class FPNOpenSeeD(BaseModel):
    def __init__(
        self,
        hidden_dim: int,
        regressor: BaseRegressor,
        resolution_level: str = "res4",
        mask_weight: float = 0.5,
        device="cuda",
    ) -> None:
        super(FPNOpenSeeD, self).__init__()
        channel_list = [96, 192, 384, 768]
        self.resolution_level = resolution_level
        self.mask_weight = mask_weight
        self.rgb_fpn = FeaturePyramidNetwork(channel_list, hidden_dim)
        self.depth_fpn = FeaturePyramidNetwork(channel_list, hidden_dim)
        self.attention = SelfAttention(hidden_dim)
        self.pooling = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(0.1)
        self.regressor = regressor
        self.openseed_seg = OpenSeeDSeg(device)

    def forward_features(
        self,
        rgb_features: dict[str, torch.Tensor],
        depth_features: dict[str, torch.Tensor],
        skip_attn: bool = False,
        **kwargs,
    ):
        B, _, _, _ = rgb_features["res2"].shape
        rgb_hidden_states: dict[str, torch.Tensor] = self.rgb_fpn(rgb_features)
        depth_hidden_states: dict[str, torch.Tensor] = self.depth_fpn(depth_features)
        _, _, fin_res_height, fin_res_width = rgb_hidden_states[
            self.resolution_level
        ].shape
        hidden_states = {}
        for (rgb_key, rgb_hid), (depth_key, depth_hid) in zip(
            rgb_hidden_states.items(), depth_hidden_states.items()
        ):
            if rgb_hid.shape[-1] > fin_res_width:
                rgb_hidden_states[rgb_key] = f.adaptive_avg_pool2d(
                    rgb_hid, (fin_res_height, fin_res_width)
                )
            else:
                rgb_hidden_states[rgb_key] = f.interpolate(
                    rgb_hid, (fin_res_height, fin_res_width)
                )
            if depth_hid.shape[-1] > fin_res_width:
                depth_hidden_states[depth_key] = f.adaptive_avg_pool2d(
                    depth_hid, (fin_res_height, fin_res_width)
                )
            else:
                depth_hidden_states[depth_key] = f.interpolate(
                    depth_hid, (fin_res_height, fin_res_width)
                )
            hidden_states[rgb_key] = (
                rgb_hidden_states[rgb_key] + depth_hidden_states[depth_key]
            )
        hidden_states = torch.stack([emb for emb in hidden_states.values()])
        if not skip_attn:
            pooled_hidden = hidden_states.mean(0)
            pooled_hidden = rearrange(
                pooled_hidden, "b c h w -> b (h w) c", h=fin_res_height
            )
            pooled_hidden = self.attention.forward(pooled_hidden)
            pooled_hidden = rearrange(
                pooled_hidden, "b (h w) c -> b c h w", h=fin_res_height
            )
            hidden_states = hidden_states + pooled_hidden.unsqueeze(0)
        hidden_states = rearrange(hidden_states, "n b c h w -> b (n c) h w", b=B)
        emb = self.pooling(hidden_states)
        emb = self.flatten(emb)
        emb = self.dropout(emb)
        out = self.regressor(emb, **kwargs)
        # for key, hid in hidden_states.items():
        #     hidden_states[key] = self.flatten(self.pooling(hid))
        # hidden_states = torch.cat([emb for emb in hidden_states.values()],dim=1)
        # out = self.regressor(hidden_states)
        return out

    def forward(self, img: torch.Tensor, depth_img: torch.Tensor, **kwargs):
        masks, _, rgb_features = self.openseed_seg.get_mask(img)
        depth_img = depth_img.expand(-1, 3, -1, -1)
        for key, feat in rgb_features.items():
            mask = f.adaptive_avg_pool2d(
                masks, (feat.shape[-2], feat.shape[-1])
            ).unsqueeze(1)
            mask = torch.where(mask.bool(), mask, self.mask_weight)
            rgb_features[key] = feat * mask
        _, _, depth_features = self.openseed_seg.get_mask(depth_img)
        for key, feat in depth_features.items():
            mask = f.adaptive_avg_pool2d(
                masks, (feat.shape[-2], feat.shape[-1])
            ).unsqueeze(1)
            mask = torch.where(mask.bool(), mask, self.mask_weight)
            depth_features[key] = feat * mask
        out = self.forward_features(rgb_features, depth_features, **kwargs)
        return out


class FPNSwin(BaseModel):
    def __init__(
        self,
        hidden_dim: int,
        pretrained_model: str = "microsoft/swin-tiny-patch4-window7-224",
        regressor: BaseRegressor = Regressor(2048, 2048),
        resolution_level: int = 2,
        mask_weight: float = 0.5,
        dropout_rate: float = 0.1,
        device="cuda",
    ) -> None:
        super(FPNSwin, self).__init__()
        channel_list = [96, 192, 384, 768]
        self.resolution_level = str(resolution_level)
        self.mask_weight = mask_weight
        self.rgb_backbone = SwinModel.from_pretrained(pretrained_model)
        self.depth_backbone = SwinModel.from_pretrained(pretrained_model)
        self.rgb_fpn = FeaturePyramidNetwork(channel_list, hidden_dim)
        self.depth_fpn = FeaturePyramidNetwork(channel_list, hidden_dim)
        self.attention = SelfAttention(hidden_dim)
        self.pooling = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(dropout_rate)
        self.regressor = regressor

    def forward(
        self,
        rgb_img: torch.Tensor,
        depth_img: torch.Tensor,
        mask: torch.Tensor,
        skip_attn: bool = False,
        **kwargs,
    ):
        B, _, H, W = rgb_img.shape
        depth_img = depth_img.expand(-1, 3, -1, -1)
        assert isinstance(self.rgb_backbone, SwinModel)
        assert isinstance(self.depth_backbone, SwinModel)
        rgb_features = self.rgb_backbone.forward(rgb_img, output_hidden_states=True)
        depth_features = self.depth_backbone.forward(
            depth_img, output_hidden_states=True
        )
        rgb_features_hid = rgb_features.hidden_states
        depth_features_hid = depth_features.hidden_states
        assert isinstance(rgb_features_hid, tuple)
        assert isinstance(depth_features_hid, tuple)
        rgb_features_dict = {}
        for i, hid in enumerate(rgb_features_hid[:-1]):
            hid = rearrange(hid, "b (h w) c -> b c h w", h=H // (2 ** (i + 2)))
            m = f.adaptive_avg_pool2d(
                mask.float(), (hid.shape[-2], hid.shape[-1])
            ).unsqueeze(1)
            m = torch.where(m.bool(), m, self.mask_weight)
            rgb_features_dict[str(i)] = hid * m
        depth_features_dict = {}
        for i, hid in enumerate(depth_features_hid[:-1]):
            hid = rearrange(hid, "b (h w) c -> b c h w", h=H // (2 ** (i + 2)))
            m = f.adaptive_avg_pool2d(
                mask.float(), (hid.shape[-2], hid.shape[-1])
            ).unsqueeze(1)
            m = torch.where(m.bool(), m, self.mask_weight)
            depth_features_dict[str(i)] = hid * m
        rgb_hidden_states = self.rgb_fpn.forward(rgb_features_dict)
        depth_hidden_states = self.depth_fpn.forward(depth_features_dict)
        _, _, fin_res_height, fin_res_width = rgb_hidden_states[
            self.resolution_level
        ].shape
        hidden_states = {}
        for (rgb_key, rgb_hid), (depth_key, depth_hid) in zip(
            rgb_hidden_states.items(), depth_hidden_states.items()
        ):
            if rgb_hid.shape[-1] > fin_res_width:
                rgb_hidden_states[rgb_key] = f.adaptive_avg_pool2d(
                    rgb_hid, (fin_res_height, fin_res_width)
                )
            else:
                rgb_hidden_states[rgb_key] = f.interpolate(
                    rgb_hid, (fin_res_height, fin_res_width)
                )
            if depth_hid.shape[-1] > fin_res_width:
                depth_hidden_states[depth_key] = f.adaptive_avg_pool2d(
                    depth_hid, (fin_res_height, fin_res_width)
                )
            else:
                depth_hidden_states[depth_key] = f.interpolate(
                    depth_hid, (fin_res_height, fin_res_width)
                )
            hidden_states[rgb_key] = (
                rgb_hidden_states[rgb_key] + depth_hidden_states[depth_key]
            )
        hidden_states = torch.stack([emb for emb in hidden_states.values()])
        if not skip_attn:
            pooled_hidden = hidden_states.mean(0)
            pooled_hidden = rearrange(
                pooled_hidden, "b c h w -> b (h w) c", h=fin_res_height
            )
            pooled_hidden = self.attention.forward(pooled_hidden)
            pooled_hidden = rearrange(
                pooled_hidden, "b (h w) c -> b c h w", h=fin_res_height
            )
            hidden_states = hidden_states + pooled_hidden.unsqueeze(0)
        hidden_states = rearrange(hidden_states, "n b c h w -> b (n c) h w", b=B)
        emb = self.pooling.forward(hidden_states)
        emb = self.flatten.forward(emb)
        emb = self.dropout.forward(emb)
        out = self.regressor(emb, **kwargs)
        return out


def get_model(config: CN, device: torch.device) -> BaseModel:
    mod = config.MODEL.NAME
    pretrained_model = config.MODEL.PRETRAINED
    layers = config.TRAIN.LAYERS
    finetune = config.TRAIN.FINETUNE
    mask_weight = config.MODEL.MASK_WEIGHT
    dropout_rate = config.MODEL.DROPOUT_RATE
    if mod == "inceptionv2":
        model = SimpleInceptionV2(
            4096,
            pretrained_model,
            regressor=Regressor(6144, 4096, dropout_rate=dropout_rate),
        )
    elif mod == "resnet50":
        model = FPN(
            512,
            pretrained_model,
            regressor=Regressor(2048, 2048, dropout_rate=dropout_rate),
        )
    elif mod == "resnet101":
        model = FPN(
            512,
            pretrained_model,
            regressor=Regressor(2048, 2048, dropout_rate=dropout_rate),
        )
    elif mod == "swin":
        model = FPNSwin(
            512,
            pretrained_model,
            regressor=Regressor(2048, 2048, dropout_rate=dropout_rate),
            mask_weight=mask_weight,
            dropout_rate=dropout_rate,
        )
    elif mod == "openseed":
        model = FPNOpenSeeD(
            512,
            regressor=Regressor(2048, 2048, dropout_rate=dropout_rate),
            mask_weight=mask_weight,
        )
    elif mod == "resnet50-ingrs":
        model = FPN(512, pretrained_model, regressor=RegressorIngrs(2048, 2048))
    elif mod == "resnet101-ingrs":
        model = FPN(512, pretrained_model, regressor=RegressorIngrs(2048, 2048))
    else:
        raise ValueError(f"Unkown model: {mod}")

    # model.backbone.requires_grad_(False)
    return model
