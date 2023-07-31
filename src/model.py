from abc import ABC, abstractmethod

import timm
import torch
import torch.nn.functional as f
import torchvision.transforms.functional as TF
from custom_utils.fpn_swin_utils import (
    AddPosEmb3D,
    AddPosEmb4D,
    AppendMask,
    DefaultMask,
    ProductMask,
    Resize2D,
    Resize3D,
)
from einops import rearrange
from seg_openseed import OpenSeeDSeg
from timm.models.vision_transformer import Attention, Block
from torch import nn
from torchvision.ops import FeaturePyramidNetwork
from transformers.models.auto.feature_extraction_auto import AutoFeatureExtractor
from transformers.models.resnet.modeling_resnet import ResNetModel
from transformers.models.swin.modeling_swin import SwinModel
from transformers.utils.generic import ModelOutput
from yacs.config import CfgNode as CN


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
        x = self.fc1(x)
        out_ingrs = self.regress_ingrs(x)
        out = {d: self.regress[d](x) for d in ["cal", "mass", "fat", "carb", "protein"]}
        out["ingrs"] = out_ingrs
        return out


class AttentionDecoder(nn.Module):
    def __init__(
        self,
        model_dim: int,
        num_layers: int,
        num_heads: int,
        mlp_ratio: float,
        dropout: float = 0.0,
    ) -> None:
        super(AttentionDecoder, self).__init__()
        self.blocks = nn.ModuleList(
            [
                Block(model_dim, num_heads, mlp_ratio, drop=dropout, attn_drop=dropout)
                for _ in range(num_layers)
            ]
        )

    def forward(self, x) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return x


class CrossAttentionBlock(nn.Module):
    def __init__(
        self, model_dim: int, num_heads: int, mlp_ratio: float, dropout: float = 0.0
    ) -> None:
        super(CrossAttentionBlock, self).__init__()
        self.layernorm1 = nn.LayerNorm(model_dim)
        self.attention1 = Attention(
            model_dim, num_heads, attn_drop=dropout, proj_drop=dropout
        )
        self.layernorm2 = nn.LayerNorm(model_dim)
        self.attention2 = Attention(
            model_dim, num_heads, attn_drop=dropout, proj_drop=dropout
        )
        mlp_dim = int(model_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(model_dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, model_dim),
        )
        self.layernorm3 = nn.LayerNorm(model_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, p, c = x.shape
        x = rearrange(x, "b n p c -> (b n) p c")
        x = self.attention1(self.layernorm1(x)) + x
        x = rearrange(x, "(b n) p c -> (b p) n c", b=b)
        x = self.attention2(self.layernorm2(x)) + x
        x = rearrange(x, "(b p) n c ->  b n p c", b=b, p=p)
        x = self.mlp(self.layernorm3(x)) + x
        return x


class CrossAttentionDecoder(nn.Module):
    def __init__(
        self,
        model_dim: int,
        num_layers: int,
        num_heads: int,
        mlp_ratio: float,
        dropout: float = 0.0,
    ) -> None:
        super(CrossAttentionDecoder, self).__init__()
        self.blocks = nn.ModuleList(
            [
                CrossAttentionBlock(model_dim, num_heads, mlp_ratio, dropout)
                for _ in range(num_layers)
            ]
        )

    def forward(self, x) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return x


class TripleCrossAttentionBlock(nn.Module):
    def __init__(
        self, model_dim: int, num_heads: int, mlp_ratio: float, dropout: float = 0.0
    ) -> None:
        super(TripleCrossAttentionBlock, self).__init__()
        self.layernorm1 = nn.LayerNorm(model_dim)
        self.attention1 = Attention(
            model_dim, num_heads, attn_drop=dropout, proj_drop=dropout
        )
        self.layernorm2 = nn.LayerNorm(model_dim)
        self.attention2 = Attention(
            model_dim, num_heads, attn_drop=dropout, proj_drop=dropout
        )
        self.layernorm3 = nn.LayerNorm(model_dim)
        self.attention3 = Attention(
            model_dim, num_heads, attn_drop=dropout, proj_drop=dropout
        )
        self.layernorm4 = nn.LayerNorm(model_dim)
        mlp_dim = int(model_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(model_dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, model_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, m, p, c = x.shape
        x = rearrange(x, "b n m p c -> (b n m) p c")
        x = self.attention1(self.layernorm1(x)) + x
        x = rearrange(x, "(b n m) p c -> (b m p) n c", b=b, n=n)
        x = self.attention2(self.layernorm2(x)) + x
        x = rearrange(x, "(b m p) n c ->  (b n p) m c", b=b, p=p)
        x = self.attention3(self.layernorm3(x)) + x
        x = rearrange(x, "(b n p) m c ->  b n m p c", b=b, p=p)
        x = self.mlp(self.layernorm4(x)) + x
        return x


class TripleCrossAttentionDecoder(nn.Module):
    def __init__(
        self,
        model_dim: int,
        num_layers: int,
        num_heads: int,
        mlp_ratio: float,
        dropout: float = 0.0,
    ) -> None:
        super(TripleCrossAttentionDecoder, self).__init__()
        self.blocks = nn.ModuleList(
            [
                TripleCrossAttentionBlock(model_dim, num_heads, mlp_ratio, dropout)
                for _ in range(num_layers)
            ]
        )

    def forward(self, x) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return x


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
        self.attention = Attention(hidden_dim)
        self.pooling = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()
        self.regressor = regressor
        for module in self.regressor.modules():
            if isinstance(module, nn.Linear):
                module.weight.data *= 0.67

    def forward(
        self,
        rgb_img: torch.Tensor,
        depth_img: torch.Tensor,
        skip_attn: bool = False,
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
        self.attention = Attention(hidden_dim)
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


class FPNSwinBase(BaseModel):
    def __init__(
        self,
        hidden_dim: int,
        pretrained_model: str = "microsoft/swin-tiny-patch4-window7-224",
        regressor: BaseRegressor = Regressor(2048, 2048),
        resolution_level: int = 2,
        mask_weight: float = 0.5,
        dropout_rate: float = 0.1,
        pos_emb: bool = False,
        fpn_after_mask: bool = True,
        device="cuda",
    ) -> None:
        super(FPNSwinBase, self).__init__()
        if pretrained_model == "microsoft/swin-tiny-patch4-window7-224":
            channel_list = [96, 192, 384, 768]
        elif pretrained_model == "microsoft/swin-large-patch4-window12-384-in22k":
            channel_list = [192, 384, 768, 1536]
        else:
            raise ValueError(pretrained_model)
        self.pos_emb = pos_emb
        self.fpn_after_mask = fpn_after_mask
        self.pretrained_model = pretrained_model
        self.init_backbone(pretrained_model)
        self.rgb_fpn = FeaturePyramidNetwork(channel_list, hidden_dim)
        self.depth_fpn = FeaturePyramidNetwork(channel_list, hidden_dim)
        self.attention = Attention(hidden_dim)
        self.pooling = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(dropout_rate)
        self.regressor = regressor
        self.mask_fn = DefaultMask(mask_weight)
        self.resize_fn = Resize2D(str(resolution_level))
        self.add_pos_emb = AddPosEmb3D()

    def init_backbone(self, pretrained_model) -> None:
        self.rgb_backbone = SwinModel.from_pretrained(pretrained_model)
        self.depth_backbone = SwinModel.from_pretrained(pretrained_model)
        if pretrained_model != "microsoft/swin-tiny-patch4-window7-224":
            self.rgb_feature_extractor = AutoFeatureExtractor.from_pretrained(
                pretrained_model
            )
            self.depth_feature_extractor = AutoFeatureExtractor.from_pretrained(
                pretrained_model
            )

    def feature_extract(
        self, rgb_img: torch.Tensor, depth_img: torch.Tensor
    ) -> tuple[tuple[torch.FloatTensor], tuple[torch.FloatTensor]]:
        depth_img = depth_img.expand(-1, 3, -1, -1)
        assert isinstance(self.rgb_backbone, SwinModel)
        assert isinstance(self.depth_backbone, SwinModel)
        if self.pretrained_model == "microsoft/swin-tiny-patch4-window7-224":
            rgb_inputs = rgb_img
            depth_inputs = depth_img
        else:
            # rgb_inputs = self.rgb_feature_extractor(images=rgb_img, return_tensors="pt")
            # depth_inputs = self.depth_feature_extractor(
            #     images=depth_img, return_tensors="pt"
            # )
            rgb_inputs = TF.center_crop(rgb_img, [384, 384])
            depth_inputs = TF.center_crop(depth_img, [384, 384])

        rgb_features = self.rgb_backbone.forward(rgb_inputs, output_hidden_states=True)  # type: ignore
        d_features = self.depth_backbone.forward(
            depth_inputs, output_hidden_states=True  # type: ignore
        )
        assert isinstance(rgb_features, ModelOutput)
        assert isinstance(d_features, ModelOutput)
        rgb_features_hid = rgb_features.reshaped_hidden_states
        d_features_hid = d_features.reshaped_hidden_states
        assert isinstance(rgb_features_hid, tuple)
        assert isinstance(d_features_hid, tuple)
        return rgb_features_hid, d_features_hid

    def attn(self, hidden_states: torch.Tensor) -> torch.Tensor:
        n, b, c, h, w = hidden_states.shape
        hidden_states_ = hidden_states.mean(0)
        hidden_states_ = rearrange(hidden_states_, "b c h w -> b (h w) c")
        hidden_states_ = self.attention.forward(hidden_states_)
        hidden_states_ = rearrange(hidden_states_, "b (h w) c -> b c h w", h=h)
        hidden_states = hidden_states + hidden_states_.unsqueeze(0)
        return hidden_states

    def forward(
        self,
        rgb_img: torch.Tensor,
        depth_img: torch.Tensor,
        mask: torch.Tensor,
        skip_attn: bool = False,
        **kwargs,
    ):
        B, _, H, W = rgb_img.shape
        rgb_features_hid, d_features_hid = self.feature_extract(rgb_img, depth_img)
        rgb_features_dict = {str(i): hid for i, hid in enumerate(rgb_features_hid[:-1])}
        d_features_dict = {str(i): hid for i, hid in enumerate(d_features_hid[:-1])}
        if self.fpn_after_mask:
            rgb_features_dict = self.mask_fn(rgb_features_dict, mask)
            d_features_dict = self.mask_fn(d_features_dict, mask)
            rgb_hidden_states = self.rgb_fpn.forward(rgb_features_dict)
            d_hidden_states = self.depth_fpn.forward(d_features_dict)
        else:
            rgb_features_dict = self.rgb_fpn.forward(rgb_features_dict)  # type: ignore
            d_features_dict = self.depth_fpn.forward(d_features_dict)  # type: ignore
            rgb_hidden_states = self.mask_fn(rgb_features_dict, mask)
            d_hidden_states = self.mask_fn(d_features_dict, mask)
        hidden_states = self.resize_fn(rgb_hidden_states, d_hidden_states)

        if self.pos_emb:
            # 3dposition embedding
            hidden_states = self.add_pos_emb(hidden_states)
        if not skip_attn:
            hidden_states = self.attn(hidden_states)
        hidden_states = rearrange(hidden_states, "n b c h w -> b (n c) h w")
        emb = self.pooling.forward(hidden_states)
        emb = self.flatten.forward(emb)
        emb = self.dropout.forward(emb)
        out = self.regressor(emb, **kwargs)
        return out


class FPNSwin(FPNSwinBase):
    def __init__(
        self,
        hidden_dim: int,
        pretrained_model: str = "microsoft/swin-tiny-patch4-window7-224",
        regressor: BaseRegressor = Regressor(2048, 2048),
        resolution_level: int = 2,
        mask_weight: float = 0.8,
        dropout_rate: float = 0.1,
        pos_emb: bool = False,
        fpn_after_mask: bool = True,
        device="cuda",
    ) -> None:
        super().__init__(
            hidden_dim,
            pretrained_model,
            regressor,
            resolution_level,
            mask_weight,
            dropout_rate,
            pos_emb,
            fpn_after_mask,
            device,
        )


class FPNCrossSwin(FPNSwinBase):
    def __init__(
        self,
        hidden_dim: int,
        num_layers: int,
        num_heads: int,
        mlp_ratio: float,
        pretrained_model: str = "microsoft/swin-tiny-patch4-window7-224",
        regressor: BaseRegressor = Regressor(3072, 3072),
        resolution_level: int = 2,
        mask_weight: float = 0.8,
        dropout_rate: float = 0.1,
        pos_emb: bool = True,
        fpn_after_mask: bool = True,
        device="cuda",
    ) -> None:
        super().__init__(
            hidden_dim,
            pretrained_model,
            regressor,
            resolution_level,
            mask_weight,
            dropout_rate,
            pos_emb,
            fpn_after_mask,
            device,
        )
        self.decoder = CrossAttentionDecoder(
            hidden_dim, num_layers, num_heads, mlp_ratio, dropout_rate
        )

    def attn(self, hidden_states: torch.Tensor) -> torch.Tensor:
        n, b, c, h, w = hidden_states.shape
        hidden_states = rearrange(hidden_states, " n b c h w -> b n (h w) c")
        hidden_states = self.decoder(hidden_states)
        hidden_states = rearrange(hidden_states, "b n (h w) c -> n b c h w", h=h)
        return hidden_states


class FPNCrossSwinCLS(FPNSwinBase):
    def __init__(
        self,
        hidden_dim: int,
        num_layers: int,
        num_heads: int,
        mlp_ratio: float,
        pretrained_model: str = "microsoft/swin-tiny-patch4-window7-224",
        regressor: BaseRegressor = Regressor(3072, 3072),
        resolution_level: int = 2,
        mask_weight: float = 0.8,
        dropout_rate: float = 0.1,
        pos_emb: bool = True,
        fpn_after_mask: bool = True,
        device="cuda",
    ) -> None:
        super().__init__(
            hidden_dim,
            pretrained_model,
            regressor,
            resolution_level,
            mask_weight,
            dropout_rate,
            pos_emb,
            fpn_after_mask,
            device,
        )
        self.decoder = CrossAttentionDecoder(
            hidden_dim, num_layers, num_heads, mlp_ratio, dropout_rate
        )
        self.cls_token = nn.Parameter(torch.zeros(hidden_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def attn(self, hidden_states: torch.Tensor) -> torch.Tensor:
        n, b, c, h, w = hidden_states.shape
        hidden_states = rearrange(hidden_states, " n b c h w -> b n (h w) c")
        cls_tokens = self.cls_token.repeat(b, n, 1, 1)
        hidden_states = torch.cat([cls_tokens, hidden_states], dim=2)
        hidden_states = self.decoder(hidden_states)
        hidden_states = hidden_states[:, :, 1:]
        hidden_states = rearrange(hidden_states, "b n (h w) c -> n b c h w", h=h)
        return hidden_states


class FPNCrossSwinMultiMask(FPNSwinBase):
    def __init__(
        self,
        hidden_dim: int,
        mask_dim: int,
        num_layers: int,
        num_heads: int,
        mlp_ratio: float,
        pretrained_model: str = "microsoft/swin-tiny-patch4-window7-224",
        regressor: BaseRegressor = Regressor(3120, 3120),
        resolution_level: int = 2,
        mask_weight: float = 0.8,
        dropout_rate: float = 0.1,
        pos_emb: bool = True,
        fpn_after_mask: bool = False,
        device="cuda",
    ) -> None:
        super().__init__(
            hidden_dim,
            pretrained_model,
            regressor,
            resolution_level,
            mask_weight,
            dropout_rate,
            pos_emb,
            fpn_after_mask,
            device,
        )
        assert mask_dim % 6 == 0
        self.decoder = CrossAttentionDecoder(
            hidden_dim + mask_dim, num_layers, num_heads, mlp_ratio, dropout_rate
        )
        self.mask_fn = AppendMask(mask_weight, mask_dim)

    def attn(self, hidden_states: torch.Tensor) -> torch.Tensor:
        n, b, c, h, w = hidden_states.shape
        hidden_states = rearrange(hidden_states, " n b c h w -> b n (h w) c")
        hidden_states = self.decoder(hidden_states)
        hidden_states = rearrange(hidden_states, "b n (h w) c -> n b c h w", h=h)
        return hidden_states


class FPNNoCrossSwinMultiMask(FPNSwinBase):
    def __init__(
        self,
        hidden_dim: int,
        mask_dim: int,
        num_layers: int,
        num_heads: int,
        mlp_ratio: float,
        pretrained_model: str = "microsoft/swin-tiny-patch4-window7-224",
        regressor: BaseRegressor = Regressor(3120, 3120),
        resolution_level: int = 2,
        mask_weight: float = 0.8,
        dropout_rate: float = 0.1,
        pos_emb: bool = True,
        fpn_after_mask: bool = False,
        device="cuda",
    ) -> None:
        super().__init__(
            hidden_dim,
            pretrained_model,
            regressor,
            resolution_level,
            mask_weight,
            dropout_rate,
            pos_emb,
            fpn_after_mask,
            device,
        )
        assert mask_dim % 6 == 0
        self.decoder = AttentionDecoder(
            hidden_dim + mask_dim, num_layers, num_heads, mlp_ratio, dropout_rate
        )
        self.mask_fn = AppendMask(mask_weight, mask_dim)

    def attn(self, hidden_states: torch.Tensor) -> torch.Tensor:
        n, b, c, h, w = hidden_states.shape
        hidden_states = rearrange(hidden_states, " n b c h w -> b (n h w) c")
        hidden_states = self.decoder(hidden_states)
        hidden_states = rearrange(hidden_states, "b (n h w) c -> n b c h w", n=n, h=h)
        return hidden_states


class FPNNoCrossSwinSingleMask(FPNSwinBase):
    def __init__(
        self,
        hidden_dim: int,
        num_layers: int,
        num_heads: int,
        mlp_ratio: float,
        pretrained_model: str = "microsoft/swin-tiny-patch4-window7-224",
        regressor: BaseRegressor = Regressor(3072, 3072),
        resolution_level: int = 2,
        mask_weight: float = 0.8,
        dropout_rate: float = 0.1,
        pos_emb: bool = True,
        fpn_after_mask: bool = False,
        device="cuda",
    ) -> None:
        super().__init__(
            hidden_dim,
            pretrained_model,
            regressor,
            resolution_level,
            mask_weight,
            dropout_rate,
            pos_emb,
            fpn_after_mask,
            device,
        )
        self.decoder = AttentionDecoder(
            hidden_dim, num_layers, num_heads, mlp_ratio, dropout_rate
        )

    def attn(self, hidden_states: torch.Tensor) -> torch.Tensor:
        n, b, c, h, w = hidden_states.shape
        hidden_states = rearrange(hidden_states, " n b c h w -> b (n h w) c")
        hidden_states = self.decoder(hidden_states)
        hidden_states = rearrange(hidden_states, "b (n h w) c -> n b c h w", n=n, h=h)
        return hidden_states


class FPNTripleCrossSwin(FPNSwinBase):
    def __init__(
        self,
        hidden_dim: int,
        mask_dim: int,
        num_layers: int,
        num_heads: int,
        mlp_ratio: float,
        pretrained_model: str = "microsoft/swin-tiny-patch4-window7-224",
        regressor: BaseRegressor = Regressor(3072, 3072),
        resolution_level: int = 2,
        mask_weight: float = 0.8,
        dropout_rate: float = 0.1,
        pos_emb: bool = True,
        fpn_after_mask: bool = False,
        device="cuda",
    ) -> None:
        super().__init__(
            hidden_dim,
            pretrained_model,
            regressor,
            resolution_level,
            mask_weight,
            dropout_rate,
            pos_emb,
            fpn_after_mask,
            device,
        )
        self.decoder = TripleCrossAttentionDecoder(
            hidden_dim, num_layers, num_heads, mlp_ratio, dropout_rate
        )
        self.mask_fn = ProductMask(mask_weight, mask_dim)
        self.resize_fn = Resize3D(str(resolution_level))
        self.add_pos_emb = AddPosEmb4D()

    def attn(self, hidden_states: torch.Tensor) -> torch.Tensor:
        n, b, m, c, h, w = hidden_states.shape
        hidden_states = rearrange(hidden_states, " n b m c h w -> b n m (h w) c")
        hidden_states = self.decoder(hidden_states)
        hidden_states = rearrange(hidden_states, "b n m (h w) c -> n b m c h w", h=h)
        hidden_states = hidden_states.mean(2)
        return hidden_states


def get_model(config: CN, device: torch.device) -> BaseModel:
    mod = config.MODEL.NAME
    pretrained_model = config.MODEL.PRETRAINED
    loss = config.TRAIN.LOSS
    dropout_rate = config.MODEL.DROPOUT_RATE
    if loss == "multi":
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
                mask_weight=config.MODEL.MASK_WEIGHT,
                dropout_rate=dropout_rate,
            )
        elif mod == "cross-swin":
            model = FPNCrossSwin(
                768,
                config.MODEL.DECODER.NUM_LAYERS,
                config.MODEL.DECODER.NUM_HEADS,
                config.MODEL.DECODER.MLP_RATIO,
                pretrained_model,
                regressor=Regressor(3072, 3072),
                mask_weight=config.MODEL.MASK_WEIGHT,
                dropout_rate=dropout_rate,
            )
        elif mod == "cross-swin-cls":
            model = FPNCrossSwinCLS(
                768,
                config.MODEL.DECODER.NUM_LAYERS,
                config.MODEL.DECODER.NUM_HEADS,
                config.MODEL.DECODER.MLP_RATIO,
                pretrained_model,
                regressor=Regressor(3072, 3072),
                mask_weight=config.MODEL.MASK_WEIGHT,
                dropout_rate=dropout_rate,
            )
        elif mod == "cross-swin-multi-mask":
            regressor_dim = 4 * (768 + config.MODEL.MASK_DIM)
            model = FPNCrossSwinMultiMask(
                768,
                config.MODEL.MASK_DIM,
                config.MODEL.DECODER.NUM_LAYERS,
                config.MODEL.DECODER.NUM_HEADS,
                config.MODEL.DECODER.MLP_RATIO,
                pretrained_model,
                regressor=Regressor(regressor_dim, regressor_dim),
                mask_weight=config.MODEL.MASK_WEIGHT,
                dropout_rate=dropout_rate,
            )
        elif mod == "no-cross-swin":
            regressor_dim = 4 * (768 + config.MODEL.MASK_DIM)
            model = FPNNoCrossSwinMultiMask(
                768,
                config.MODEL.MASK_DIM,
                config.MODEL.DECODER.NUM_LAYERS,
                config.MODEL.DECODER.NUM_HEADS,
                config.MODEL.DECODER.MLP_RATIO,
                pretrained_model,
                regressor=Regressor(regressor_dim, regressor_dim),
                mask_weight=config.MODEL.MASK_WEIGHT,
                dropout_rate=dropout_rate,
            )
        elif mod == "no-cross-swin-single-mask":
            model = FPNNoCrossSwinSingleMask(
                768,
                config.MODEL.DECODER.NUM_LAYERS,
                config.MODEL.DECODER.NUM_HEADS,
                config.MODEL.DECODER.MLP_RATIO,
                pretrained_model,
                regressor=Regressor(3072, 3072),
                mask_weight=config.MODEL.MASK_WEIGHT,
                dropout_rate=dropout_rate,
            )
        elif mod == "triple-cross-swin":
            model = FPNTripleCrossSwin(
                768,
                config.MODEL.MASK_DIM,
                config.MODEL.DECODER.NUM_LAYERS,
                config.MODEL.DECODER.NUM_HEADS,
                config.MODEL.DECODER.MLP_RATIO,
                pretrained_model,
                regressor=Regressor(3072, 3072),
                mask_weight=config.MODEL.MASK_WEIGHT,
                dropout_rate=dropout_rate,
            )
        elif mod == "openseed":
            model = FPNOpenSeeD(
                512,
                regressor=Regressor(2048, 2048, dropout_rate=dropout_rate),
                mask_weight=config.MODEL.MASK_WEIGHT,
            )
        else:
            raise ValueError(f"Unkown model and loss: {mod} and {loss}")
    elif loss == "multi_ingrs":
        if mod == "resnet50-ingrs":
            model = FPN(512, pretrained_model, regressor=RegressorIngrs(2048, 2048))
        elif mod == "resnet101-ingrs":
            model = FPN(512, pretrained_model, regressor=RegressorIngrs(2048, 2048))
        elif mod == "swin":
            model = FPNSwin(
                512,
                pretrained_model,
                regressor=RegressorIngrs(2048, 2048),
                mask_weight=config.MODEL.MASK_WEIGHT,
                dropout_rate=dropout_rate,
            )
        elif mod == "cross-swin":
            model = FPNCrossSwin(
                768,
                config.MODEL.DECODER.NUM_LAYERS,
                config.MODEL.DECODER.NUM_HEADS,
                config.MODEL.DECODER.MLP_RATIO,
                pretrained_model,
                regressor=RegressorIngrs(3072, 3072),
                mask_weight=config.MODEL.MASK_WEIGHT,
                dropout_rate=dropout_rate,
            )
        elif mod == "cross-swin-cls":
            model = FPNCrossSwinCLS(
                768,
                config.MODEL.DECODER.NUM_LAYERS,
                config.MODEL.DECODER.NUM_HEADS,
                config.MODEL.DECODER.MLP_RATIO,
                pretrained_model,
                regressor=RegressorIngrs(3072, 3072),
                mask_weight=config.MODEL.MASK_WEIGHT,
                dropout_rate=dropout_rate,
            )
        elif mod == "no-cross-swin":
            regressor_dim = 4 * (768 + config.MODEL.MASK_DIM)
            model = FPNNoCrossSwinMultiMask(
                768,
                config.MODEL.MASK_DIM,
                config.MODEL.DECODER.NUM_LAYERS,
                config.MODEL.DECODER.NUM_HEADS,
                config.MODEL.DECODER.MLP_RATIO,
                pretrained_model,
                regressor=RegressorIngrs(regressor_dim, regressor_dim),
                mask_weight=config.MODEL.MASK_WEIGHT,
                dropout_rate=dropout_rate,
            )
        elif mod == "no-cross-swin-single-mask":
            model = FPNNoCrossSwinSingleMask(
                768,
                config.MODEL.DECODER.NUM_LAYERS,
                config.MODEL.DECODER.NUM_HEADS,
                config.MODEL.DECODER.MLP_RATIO,
                pretrained_model,
                regressor=RegressorIngrs(3072, 3072),
                mask_weight=config.MODEL.MASK_WEIGHT,
                dropout_rate=dropout_rate,
            )
        elif mod == "triple-cross-swin":
            model = FPNTripleCrossSwin(
                768,
                config.MODEL.MASK_DIM,
                config.MODEL.DECODER.NUM_LAYERS,
                config.MODEL.DECODER.NUM_HEADS,
                config.MODEL.DECODER.MLP_RATIO,
                pretrained_model,
                regressor=RegressorIngrs(3072, 3072),
                mask_weight=config.MODEL.MASK_WEIGHT,
                dropout_rate=dropout_rate,
            )
        elif mod == "cross-swin-multi-mask":
            regressor_dim = 4 * (768 + config.MODEL.MASK_DIM)
            model = FPNCrossSwinMultiMask(
                768,
                config.MODEL.MASK_DIM,
                config.MODEL.DECODER.NUM_LAYERS,
                config.MODEL.DECODER.NUM_HEADS,
                config.MODEL.DECODER.MLP_RATIO,
                pretrained_model,
                regressor=RegressorIngrs(regressor_dim, regressor_dim),
                mask_weight=config.MODEL.MASK_WEIGHT,
                dropout_rate=dropout_rate,
            )
        else:
            raise ValueError(f"Unkown model and loss: {mod} and {loss}")
    else:
        raise ValueError(f"Unkown model and loss: {mod} and {loss}")
    return model
