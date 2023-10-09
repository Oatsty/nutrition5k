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
    """
    Nutrient Regressor. Output a dictionary of the predicted nutritional values (calorie, mass, fat, carb, and protein).
    Consist of one shared linear layer and 1 or 2 separated linear layers.

    Args:
        in_dim (int): Dimension of the input embeddings
        hidden_dim (int): Hidden dimension
    """
    def __init__(self, in_dim: int, hidden_dim: int) -> None:
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
                    nn.Linear(hidden_dim, 1),
                )
                for x in ["cal", "mass", "fat", "carb", "protein"]
            }
        )

    def forward(self, x: torch.Tensor, depth: int = 3):
        """
        Compute predicted nutritional values.

        Args:
            x (torch.Tensor): input embeddings
            depth (int): linear layer depth. (default=3)
        Return:
            dict[str, torch.Tensor] : output nutritional values (cal, mass, fat, carb, protein)
        """
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
    """
    Nutrient and Ingredient Regressor. Output a dictionary of the predicted nutritional values (calorie, mass, fat, carb, and protein) and ingredients.
    Nutrition-5k consists of a total of 555 different ingredients.
    Consist of one shared linear layer and 2 separated linear layers.

    Args:
        in_dim (int): Dimension of the input embeddings
        hidden_dim (int): Hidden dimension
    """
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
        """
        Compute predicted nutritional values and ingredients.

        Args:
            x (torch.Tensor): input embeddings
            depth (int): linear layer depth. (default=3)
        Return:
            dict[str, torch.Tensor] : output nutritional values (cal, mass, fat, carb, protein) and ingredients
        """
        x = self.fc1(x)
        out_ingrs = self.regress_ingrs(x)
        out = {d: self.regress[d](x) for d in ["cal", "mass", "fat", "carb", "protein"]}
        out["ingrs"] = out_ingrs
        return out


class AttentionDecoder(nn.Module):
    """
    Simple Transformer Decoder with one self-attention mechanism per block.

    Args:
        model_dim (int): model dimension (qkv dimension)
        num_layers (int): number of transformer blocks
        num_heads (int): numer of heads for multihead attention
        mlp_ratio (float): ratio of mlp hidden dimension to model dimension
        dropout (float): drop out rate. (default = 0.0)
    """
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
        """
        Compute simple transformer decoder.

        Args:
            x (torch.Tensor): input embeddings [batch x position x channel]
        Return:
            torch.Tensor: output embeddings (same shape as input)
        """
        for block in self.blocks:
            x = block(x)
        return x


class CrossAttentionBlock(nn.Module):
    """
    Cross-Attention Transformer Block with two self-attention mechanisms (layer-wise and position-wise) per block.

    Args:
        model_dim (int): model dimension (qkv dimension)
        num_heads (int): numer of heads for multihead attention
        mlp_ratio (float): ratio of mlp hidden dimension to model dimension
        dropout (float): drop out rate. (default = 0.0)
    """
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
        """
        Compute one cross-attention transformer block.

        Args:
            x (torch.Tensor): input embeddings [batch x num_layer x position x channel]
        Return:
            torch.Tensor: output embeddings (same shape as input)
        """
        b, n, p, c = x.shape
        x = rearrange(x, "b n p c -> (b n) p c")
        x = self.attention1(self.layernorm1(x)) + x
        x = rearrange(x, "(b n) p c -> (b p) n c", b=b)
        x = self.attention2(self.layernorm2(x)) + x
        x = rearrange(x, "(b p) n c ->  b n p c", b=b, p=p)
        x = self.mlp(self.layernorm3(x)) + x
        return x


class CrossAttentionDecoder(nn.Module):
    """
    Cross-Attention Transformer Decoder with two self-attention mechanisms (layer-wise and position-wise) per block.

    Args:
        model_dim (int): model dimension (qkv dimension)
        num_layers (int): number of transformer blocks
        num_heads (int): numer of heads for multihead attention
        mlp_ratio (float): ratio of mlp hidden dimension to model dimension
        dropout (float): drop out rate. (default = 0.0)
    """
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
        """
        Compute cross-attention transformer decoder.

        Args:
            x (torch.Tensor): input embeddings [batch x num_layer x position x channel]
        Return:
            torch.Tensor: output embeddings (same shape as input)
        """
        for block in self.blocks:
            x = block(x)
        return x


class TripleCrossAttentionBlock(nn.Module):
    """
    Triple-Cross-Attention Transformer Block with three self-attention mechanisms (layer-wise, position-wise, and mask-wise) per block.

    Args:
        model_dim (int): model dimension (qkv dimension)
        num_heads (int): numer of heads for multihead attention
        mlp_ratio (float): ratio of mlp hidden dimension to model dimension
        dropout (float): drop out rate. (default = 0.0)
    """
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
        """
        Compute one triple-cross-attention transformer block.

        Args:
            x (torch.Tensor): input embeddings [batch x num_layer x num_mask x position x channel]
        Return:
            torch.Tensor: output embeddings (same shape as input)
        """
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
    """
    Triple-Cross-Attention Transformer Decoder with three self-attention mechanisms (layer-wise, position-wise, and mask-wise) per block.

    Args:
        model_dim (int): model dimension (qkv dimension)
        num_layers (int): number of transformer blocks
        num_heads (int): numer of heads for multihead attention
        mlp_ratio (float): ratio of mlp hidden dimension to model dimension
        dropout (float): drop out rate. (default = 0.0)
    """
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
        """
        Compute triple-cross-attention transformer decoder.

        Args:
            x (torch.Tensor): input embeddings [batch x num_layer x num_mask x position x channel]
        Return:
            torch.Tensor: output embeddings (same shape as input)
        """
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
    """
    Google-Nutrition model implementation.

    Args:
        hidden_dim (int): hidden dimension
        pretrained_model (str): pretrained model name (default = inception_resnet_v2)
        regressor (BaseRegressor): Nutrient Regressor (default = Regrerssor(6144, 4096))
    """
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
        """
        Google-Nutrition implementation.

        Args:
            img (torch.Tensor): rgb image tensor [B x C x H x W]
            depth (torch.Tensor): depth image tensor [B x 1 x H x W]
        Return:
            dict[str, torch.Tensor]: output nutritional values (cal, mass, fat, carb, protein)
        """
        x = torch.cat([img, depth], dim=1)
        x = self.conv0(x)
        x = self.backbone(x)[-1]
        x = self.pooling(x)
        x = self.flatten(x)
        out = self.regressor(x)
        return out


class FPN(BaseModel):
    """
    MMFF-Nutrition model implementation.

    Args:
        hidden_dim (int): hidden dimension for attention, FPN, etc.
        pretrained_model (str): pretrained model name (default = microsoft/resnet-50)
        resolution_level (int): resolution level to interpolate after FPN (default=2)
        regressor (BaseRegressor): Nutrient Regressor (default = Regressor(2048, 2048))
    """
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
        """
        MMFF-Nutrition implementation

        Args:
            rgb_img (torch.Tensor): rgb image tensor [B x C x H x W]
            depth_img (torch.Tensor): depth image tensor [B x 1 x H x W]
            skip_attn (bool): whether to skip attetion mechanism (default=False)
        Return:
            dict[str, torch.Tensor]: output nutritional values (cal, mass, fat, carb, protein)
        """
        B, _, _, _ = rgb_img.shape
        depth_img = depth_img.expand(-1, 3, -1, -1)
        inputs = {"rgb_img": rgb_img, "depth_img": depth_img}

        #extract embeddings from backbone
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

        #FPN
        hidden_states = self.fpn(hidden_states_dict)

        #Resize to specified resolution level
        fin_res = hidden_states[self.resolution_level].shape[-1]
        for i, hid in hidden_states.items():
            if i <= self.resolution_level:
                hidden_states[i] = f.adaptive_avg_pool2d(hid, fin_res)
            else:
                hidden_states[i] = f.interpolate(hid, fin_res)
        hidden_states = torch.stack([emb for emb in hidden_states.values()])

        # self-attention
        if not skip_attn:
            pooled_hidden = hidden_states.mean(0)
            pooled_hidden = rearrange(pooled_hidden, "b c h w -> b (h w) c", h=fin_res)
            pooled_hidden = self.attention(pooled_hidden)
            pooled_hidden = rearrange(pooled_hidden, "b (h w) c -> b c h w", h=fin_res)
            hidden_states = hidden_states + pooled_hidden.unsqueeze(0)
        hidden_states = rearrange(hidden_states, "n b c h w -> b (n c) h w", b=B)

        #nutrient regression
        emb = self.pooling(hidden_states)
        emb = self.flatten(emb)
        out = self.regressor(emb)
        return out


class FPNSwinBase(BaseModel):
    """
    Base model for our implementation.

    Args:
        hidden_dim (int): hidden dimension for FPN, attention, etc.
        pretrained_model (str): pretrained model name (default = microsoft/swin-tiny-patch4-window7-224)
        regressor (BaseRegressor): nutrient regressor (default = Regressor(2048, 2048))
        resolution_level (int): resolution level to interpolate after FPN (default=2)
        mask_weight (float): mask_ratio [0,1] (default=0.5)
        dropout_rate (float): drop-out rate [0,1] (default=0.1)
        pos_emb (bool): whether to perform positional embedding (default=False)
        fpn_after_mask (bool): whether to apply food region mask on the embeddings after or before FPN (default=True)
        device (torch.device): device (default=cuda)
    """
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
        """
        Init model backbone

        Args:
            pretrained_model (str): pretrained model name
        """
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
        """
        Extract feature from model backbone

        Args:
            rgb_img (torch.Tensor): rgb image tensor [B x C x H x W]
            depth_img (torch.Tensor): depth image tensor [B x 1 x H x W]
        Return:
            rgb_features_hid, depth_features_hid (tuple[tuple[torch.FloatTensor], tuple[torch.FloatTensor]]):
            Each features_hid is a tuple of hidden embeddings from all layers of the backbone model. [B x C' x H' x W']
        """
        depth_img = depth_img.expand(-1, 3, -1, -1)
        assert isinstance(self.rgb_backbone, SwinModel)
        assert isinstance(self.depth_backbone, SwinModel)
        if self.pretrained_model == "microsoft/swin-tiny-patch4-window7-224":
            rgb_inputs = rgb_img
            depth_inputs = depth_img
        else:
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
        """
        Perform self-attention or Trasnformer decoder

        Args:
            hidden_states (torch.Tensor): input embeddings [B x C x H x W]
        Return:
            torch.Tensor: output embeddings [B x C x H x W]
        """
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
        """
        Our baseline implementation

        Args:
            rgb_img (torch.Tensor): rgb image tensor [B x C x H x W]
            depth_img (torch.Tensor): depth image tensor [B x 1 x H x W]
            mask (torch.Tensor): food region mask tensor [B x (M) x H x W]
            skip_attn (bool): whether to skip attetion mechanism (default=False)
        Return:
            dict[str, torch.Tensor]: output nutritional values (cal, mass, fat, carb, protein)
        """
        B, _, H, W = rgb_img.shape

        # extract feature from backbone
        rgb_features_hid, d_features_hid = self.feature_extract(rgb_img, depth_img)
        rgb_features_dict = {str(i): hid for i, hid in enumerate(rgb_features_hid[:-1])}
        d_features_dict = {str(i): hid for i, hid in enumerate(d_features_hid[:-1])}

        # compute FPN and apply food region mask
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

        # postional embedding
        if self.pos_emb:
            hidden_states = self.add_pos_emb(hidden_states)

        # self-attention or Transformer decoder
        if not skip_attn:
            hidden_states = self.attn(hidden_states)
        hidden_states = rearrange(hidden_states, "n b c h w -> b (n c) h w")

        # Regression
        emb = self.pooling.forward(hidden_states)
        emb = self.flatten.forward(emb)
        emb = self.dropout.forward(emb)
        out = self.regressor(emb, **kwargs)
        return out


class FPNSwin(FPNSwinBase):
    """
    Baseline model for our implementation.

    Args:
        hidden_dim (int): hidden dimension for FPN, attention, etc.
        pretrained_model (str): pretrained model name (default = microsoft/swin-tiny-patch4-window7-224)
        regressor (BaseRegressor): nutrient regressor (default = Regressor(2048, 2048))
        resolution_level (int): resolution level to interpolate after FPN (default=2)
        mask_weight (float): mask_ratio [0,1] (default=0.8)
        dropout_rate (float): drop-out rate [0,1] (default=0.1)
        pos_emb (bool): whether to perform positional embedding (default=False)
        fpn_after_mask (bool): whether to apply food region mask on the embeddings after or before FPN (default=True)
        device (torch.device): device (default=cuda)
    """
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
    """
    Our model with cross-attention decoder without class token.

    Args:
        hidden_dim (int): hidden dimension for FPN, attention, etc.
        num_layers (int): number of transformer blocks
        num_heads (int): numer of heads for multihead attention
        mlp_ratio (float): ratio of mlp hidden dimension to model dimension
        pretrained_model (str): pretrained model name (default = microsoft/swin-tiny-patch4-window7-224)
        regressor (BaseRegressor): nutrient regressor (default = Regressor(3072, 3072))
        resolution_level (int): resolution level to interpolate after FPN (default=2)
        mask_weight (float): mask_ratio [0,1] (default=0.8)
        dropout_rate (float): drop-out rate [0,1] (default=0.1)
        pos_emb (bool): whether to perform positional embedding (default=True)
        fpn_after_mask (bool): whether to apply food region mask on the embeddings after or before FPN (default=True)
        device (torch.device): device (default=cuda)
    """
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
        """
        Compute using cross-attention transformer decoder

        Args:
            hidden_states (torch.Tensor): input embeddings [N x B x C x H x W]
        Return:
            torch.Tensor: output embeddings [N x B x C x H x W]
        """
        n, b, c, h, w = hidden_states.shape
        hidden_states = rearrange(hidden_states, " n b c h w -> b n (h w) c")
        hidden_states = self.decoder(hidden_states)
        hidden_states = rearrange(hidden_states, "b n (h w) c -> n b c h w", h=h)
        return hidden_states


class FPNCrossSwinCLS(FPNSwinBase):
    """
    Our model with cross-attention decoder with class token.

    Args:
        hidden_dim (int): hidden dimension for FPN, attention, etc.
        num_layers (int): number of transformer blocks
        num_heads (int): numer of heads for multihead attention
        mlp_ratio (float): ratio of mlp hidden dimension to model dimension
        pretrained_model (str): pretrained model name (default = microsoft/swin-tiny-patch4-window7-224)
        regressor (BaseRegressor): nutrient regressor (default = Regressor(3072, 3072))
        resolution_level (int): resolution level to interpolate after FPN (default=2)
        mask_weight (float): mask_ratio [0,1] (default=0.8)
        dropout_rate (float): drop-out rate [0,1] (default=0.1)
        pos_emb (bool): whether to perform positional embedding (default=True)
        fpn_after_mask (bool): whether to apply food region mask on the embeddings after or before FPN (default=True)
        device (torch.device): device (default=cuda)
    """
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
        """
        Compute using cross-attention transformer decoder

        Args:
            hidden_states (torch.Tensor): input embeddings [N x B x C x H x W]
        Return:
            torch.Tensor: output embeddings [N x B x C x H x W]
        """
        n, b, c, h, w = hidden_states.shape
        hidden_states = rearrange(hidden_states, " n b c h w -> b n (h w) c")
        cls_tokens = self.cls_token.repeat(b, n, 1, 1)
        hidden_states = torch.cat([cls_tokens, hidden_states], dim=2)
        hidden_states = self.decoder(hidden_states)
        hidden_states = hidden_states[:, :, 1:]
        hidden_states = rearrange(hidden_states, "b n (h w) c -> n b c h w", h=h)
        return hidden_states


class FPNCrossSwinMultiMask(FPNSwinBase):
    """
    Our model with cross-attention decoder and multi-masks. Multiple masks are apply by concatenating with hidden embeddings

    Args:
        hidden_dim (int): hidden dimension for FPN, attention, etc.
        mask_dim (int): number of masks. For an easier implementation, mask_dim should be a multiple of 6
        num_layers (int): number of transformer blocks
        num_heads (int): numer of heads for multihead attention
        mlp_ratio (float): ratio of mlp hidden dimension to model dimension
        pretrained_model (str): pretrained model name (default = microsoft/swin-tiny-patch4-window7-224)
        regressor (BaseRegressor): nutrient regressor (default = Regressor(3120, 3120))
        resolution_level (int): resolution level to interpolate after FPN (default=2)
        mask_weight (float): mask_ratio [0,1] (default=0.8)
        dropout_rate (float): drop-out rate [0,1] (default=0.1)
        pos_emb (bool): whether to perform positional embedding (default=True)
        fpn_after_mask (bool): whether to apply food region mask on the embeddings after or before FPN (default=False)
        device (torch.device): device (default=cuda)
    """
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
        """
        Compute using cross-attention transformer decoder

        Args:
            hidden_states (torch.Tensor): input embeddings [N x B x C x H x W]
        Return:
            torch.Tensor: output embeddings [N x B x C x H x W]
        """
        n, b, c, h, w = hidden_states.shape
        hidden_states = rearrange(hidden_states, " n b c h w -> b n (h w) c")
        hidden_states = self.decoder(hidden_states)
        hidden_states = rearrange(hidden_states, "b n (h w) c -> n b c h w", h=h)
        return hidden_states


class FPNNoCrossSwinMultiMask(FPNSwinBase):
    """
    Our model with simple attention decoder and multi-masks. Multiple masks are apply by concatenating with hidden embeddings

    Args:
        hidden_dim (int): hidden dimension for FPN, attention, etc.
        mask_dim (int): number of masks. For an easier implementation, mask_dim should be a multiple of 6
        num_layers (int): number of transformer blocks
        num_heads (int): numer of heads for multihead attention
        mlp_ratio (float): ratio of mlp hidden dimension to model dimension
        pretrained_model (str): pretrained model name (default = microsoft/swin-tiny-patch4-window7-224)
        regressor (BaseRegressor): nutrient regressor (default = Regressor(3120, 3120))
        resolution_level (int): resolution level to interpolate after FPN (default=2)
        mask_weight (float): mask_ratio [0,1] (default=0.8)
        dropout_rate (float): drop-out rate [0,1] (default=0.1)
        pos_emb (bool): whether to perform positional embedding (default=True)
        fpn_after_mask (bool): whether to apply food region mask on the embeddings after or before FPN (default=False)
        device (torch.device): device (default=cuda)
    """
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
        """
        Compute using simple self-attention transformer decoder

        Args:
            hidden_states (torch.Tensor): input embeddings [N x B x C x H x W]
        Return:
            torch.Tensor: output embeddings [N x B x C x H x W]
        """
        n, b, c, h, w = hidden_states.shape
        hidden_states = rearrange(hidden_states, " n b c h w -> b (n h w) c")
        hidden_states = self.decoder(hidden_states)
        hidden_states = rearrange(hidden_states, "b (n h w) c -> n b c h w", n=n, h=h)
        return hidden_states


class FPNNoCrossSwinSingleMask(FPNSwinBase):
    """
    Our model with simple attention decoder and single mask.

    Args:
        hidden_dim (int): hidden dimension for FPN, attention, etc.
        num_layers (int): number of transformer blocks
        num_heads (int): numer of heads for multihead attention
        mlp_ratio (float): ratio of mlp hidden dimension to model dimension
        pretrained_model (str): pretrained model name (default = microsoft/swin-tiny-patch4-window7-224)
        regressor (BaseRegressor): nutrient regressor (default = Regressor(3072, 3072))
        resolution_level (int): resolution level to interpolate after FPN (default=2)
        mask_weight (float): mask_ratio [0,1] (default=0.8)
        dropout_rate (float): drop-out rate [0,1] (default=0.1)
        pos_emb (bool): whether to perform positional embedding (default=True)
        fpn_after_mask (bool): whether to apply food region mask on the embeddings after or before FPN (default=False)
        device (torch.device): device (default=cuda)
    """
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
        """
        Compute using simple self-attention transformer decoder

        Args:
            hidden_states (torch.Tensor): input embeddings [N x B x C x H x W]
        Return:
            torch.Tensor: output embeddings [N x B x C x H x W]
        """
        n, b, c, h, w = hidden_states.shape
        hidden_states = rearrange(hidden_states, " n b c h w -> b (n h w) c")
        hidden_states = self.decoder(hidden_states)
        hidden_states = rearrange(hidden_states, "b (n h w) c -> n b c h w", n=n, h=h)
        return hidden_states


class FPNTripleCrossSwin(FPNSwinBase):
    """
    Our model with triple-cross-attention decoder.

    Args:
        hidden_dim (int): hidden dimension for FPN, attention, etc.
        mask_dim (int): number of masks.
        num_layers (int): number of transformer blocks
        num_heads (int): numer of heads for multihead attention
        mlp_ratio (float): ratio of mlp hidden dimension to model dimension
        pretrained_model (str): pretrained model name (default = microsoft/swin-tiny-patch4-window7-224)
        regressor (BaseRegressor): nutrient regressor (default = Regressor(3072, 3072))
        resolution_level (int): resolution level to interpolate after FPN (default=2)
        mask_weight (float): mask_ratio [0,1] (default=0.8)
        dropout_rate (float): drop-out rate [0,1] (default=0.1)
        pos_emb (bool): whether to perform positional embedding (default=True)
        fpn_after_mask (bool): whether to apply food region mask on the embeddings after or before FPN (default=False)
        device (torch.device): device (default=cuda)
    """
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
        """
        Compute using triple-cross-attention transformer decoder

        Args:
            hidden_states (torch.Tensor): input embeddings [N x B x M x C x H x W]
        Return:
            torch.Tensor: output embeddings [N x B x M x C x H x W]
        """
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
                regressor=Regressor(6144, 4096),
            )
        elif mod == "resnet50":
            model = FPN(
                512,
                pretrained_model,
                regressor=Regressor(2048, 2048),
            )
        elif mod == "resnet101":
            model = FPN(
                512,
                pretrained_model,
                regressor=Regressor(2048, 2048),
            )
        elif mod == "swin":
            model = FPNSwin(
                512,
                pretrained_model,
                regressor=Regressor(2048, 2048),
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
