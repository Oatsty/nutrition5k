from abc import ABC, abstractmethod

import torch
import torch.nn.functional as f
from einops import rearrange
from torch import nn

from .position_embedding import get_pos_embed_nd


class Mask(nn.Module, ABC):
    """
    An abstract class for generating embeddings from hidden features and food region masks

    Args:
        mask_weight (float): mask_ratio
    """
    def __init__(self, mask_weight: float) -> None:
        super().__init__()
        self.mask_weight = mask_weight

    @abstractmethod
    def __call__(
        self,
        feats_hid: dict[str, torch.Tensor] | dict[str, torch.FloatTensor],
        mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        Generate embeddings from hidden features and food region masks

        Args:
            feats_hid (dict[str, torch.Tensor]): Dictionary of hidden features
            mask (torch.Tensor): food region binary mask
        Return:
            dict[str, torch.Tensor]: The output embeddings.
        """
        ...


class DefaultMask(Mask):
    def __call__(
        self,
        feats_hid: dict[str, torch.Tensor] | dict[str, torch.FloatTensor],
        mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        Generate embeddings from hidden features and food region masks by multiplying the background region with mask_weight

        Args:
            feats_hid (dict[str, torch.Tensor]): Dictionary of hidden features. Shape [B x C x H x W]
            mask (torch.Tensor): food region binary mask with shape [B x mask_number x H x W] or [B x H x W]
        Return:
            dict[str, torch.Tensor]: The output embeddings. Have same keys and shape as feats_hid.
        """
        if len(mask.shape) > 4 or len(mask.shape) <= 2:
            raise ValueError(f"Invalid mask shape: {mask.shape}")
        if len(mask.shape) == 4:
            mask = mask.mean(1)
        feats_dict = {}
        for i, hid in feats_hid.items():
            m = f.adaptive_avg_pool2d(
                mask.float(), (hid.shape[-2], hid.shape[-1])
            ).unsqueeze(1)
            m = torch.where(m.bool(), m, self.mask_weight)
            feats_dict[i] = hid * m
        return feats_dict


class AppendMask(Mask):
    """
    An abstract class for generating embeddings from hidden features and food region masks

    Args:
        mask_weight (float): mask_ratio
        mask_dim (int): highest number of masks that can be concatenate
    """
    def __init__(self, mask_weight: float, mask_dim: int) -> None:
        super().__init__(mask_weight)
        self.mask_dim = mask_dim

    def __call__(
        self,
        feats_hid: dict[str, torch.FloatTensor] | dict[str, torch.Tensor],
        mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        Generate embeddings from hidden features and food region masks by concatenating the food mask on top of the original features

        Args:
            feats_hid (dict[str, torch.Tensor]): Dictionary of hidden features. Shape [B x C x H x W]
            mask (torch.Tensor): food region binary mask with shape [B x mask_number x H x W] or [B x H x W]
        Return:
            dict[str, torch.Tensor]: The output embeddings. Have same keys as feats_hid. Shape is [B x (C + mask_dim) x H x W]
        """
        assert (
            mask.shape[1] >= self.mask_dim
        ), f"mask shape[1]: {mask.shape[1]} less than mask_dim"
        if len(mask.shape) > 4 or len(mask.shape) <= 2:
            raise ValueError(f"Invalid mask shape: {mask.shape}")
        if len(mask.shape) == 3:
            mask = mask.unsqueeze(1)
        feats_dict = {}
        for i, hid in feats_hid.items():
            m = f.adaptive_avg_pool2d(mask.float(), (hid.shape[-2], hid.shape[-1]))
            feats_dict[i] = torch.cat([hid, m[:, : self.mask_dim]], dim=1)
        return feats_dict


class ProductMask(Mask):
    """
    An abstract class for generating embeddings from hidden features and food region masks

    Args:
        mask_weight (float): mask_ratio
        mask_dim (int): highest number of masks that can be used
    """
    def __init__(self, mask_weight: float, mask_dim: int) -> None:
        super().__init__(mask_weight)
        self.mask_dim = mask_dim

    def __call__(
        self,
        feats_hid: dict[str, torch.FloatTensor] | dict[str, torch.Tensor],
        mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        Generate embeddings from hidden features and food region masks by multiplying all food masks with the original features

        Args:
            feats_hid (dict[str, torch.Tensor]): Dictionary of hidden features. Shape [B x C x H x W]
            mask (torch.Tensor): food region binary mask with shape [B x mask_number x H x W] or [B x H x W]
        Return:
            dict[str, torch.Tensor]: The output embeddings. Have same keys as feats_hid. Shape [B x mask_dim x C x H x W]
        """
        if len(mask.shape) > 4 or len(mask.shape) <= 2:
            raise ValueError(f"Invalid mask shape: {mask.shape}")
        if len(mask.shape) == 3:
            mask = mask.unsqueeze(1)
        feats_dict = {}
        for i, hid in feats_hid.items():
            m = f.adaptive_avg_pool2d(mask.float(), (hid.shape[-2], hid.shape[-1]))[
                :, : self.mask_dim
            ]
            m = torch.where(m.bool(), m, self.mask_weight)
            feats_dict[i] = torch.einsum("bchw, bmhw -> bmchw", hid, m)
        return feats_dict


class Resize(nn.Module, ABC):
    """
    Perform resizing and element-wise summation of RGB and depth embeddings.

    Args:
        resolution_level (str): The key of the layer that will be used as the final resolution level
    """
    def __init__(self, resolution_level: str) -> None:
        super().__init__()
        self.resolution_level = resolution_level

    @abstractmethod
    def __call__(
        self,
        rgb_hids: dict[str, torch.Tensor],
        d_hids: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        Perform resizing and element-wise summation of RGB and depth embeddings.

        Args:
            rgb_hids dict[str, torch.Tensor]: RGB embeddings
            d_hids dict[str, torch.Tensor]: depth embeddings
        Return:
            torch.Tensor: output embeddings
        """
        ...


class Resize2D(Resize):
    def __call__(
        self,
        rgb_hids: dict[str, torch.Tensor],
        d_hids: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        Perform resizing and element-wise summation of RGB and depth embeddings with [B x C x H x W] shape.

        Args:
            rgb_hids dict[str, torch.Tensor]: RGB embeddings. [B x C x H x W]
            d_hids dict[str, torch.Tensor]: depth embeddings. [B x C x H x W]
        Return:
            torch.Tensor: output embeddings containing every layer [N_layer x B x C x H x W]
        """
        _, _, fin_res_h, fin_res_w = rgb_hids[self.resolution_level].shape
        hidden_states = {}
        for (rgb_k, rgb_hid), (d_k, d_hid) in zip(rgb_hids.items(), d_hids.items()):
            if rgb_hid.shape[-1] > fin_res_w:
                rgb_hids[rgb_k] = f.adaptive_avg_pool2d(rgb_hid, (fin_res_h, fin_res_w))
            else:
                rgb_hids[rgb_k] = f.interpolate(rgb_hid, (fin_res_h, fin_res_w))
            if d_hid.shape[-1] > fin_res_w:
                d_hids[d_k] = f.adaptive_avg_pool2d(d_hid, (fin_res_h, fin_res_w))
            else:
                d_hids[d_k] = f.interpolate(d_hid, (fin_res_h, fin_res_w))
            hidden_states[rgb_k] = rgb_hids[rgb_k] + d_hids[d_k]
        hidden_states = torch.stack([emb for emb in hidden_states.values()])
        return hidden_states


class Resize3D(Resize):
    def __call__(
        self,
        rgb_hids: dict[str, torch.Tensor],
        d_hids: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        Perform resizing and element-wise summation of RGB and depth embeddings with [B x M X C x H x W] shape.

        Args:
            rgb_hids dict[str, torch.Tensor]: RGB embeddings. [B x M X C x H x W]
            d_hids dict[str, torch.Tensor]: depth embeddings. [B x M X C x H x W]
        Return:
            torch.Tensor: output embeddings containing every layer [N_layer x B x C x H x W]
        """
        _, _, C, fin_res_h, fin_res_w = rgb_hids[self.resolution_level].shape
        hidden_states = {}
        for (rgb_k, rgb_hid), (d_k, d_hid) in zip(rgb_hids.items(), d_hids.items()):
            if rgb_hid.shape[-1] > fin_res_w:
                rgb_hids[rgb_k] = f.adaptive_avg_pool3d(
                    rgb_hid, (C, fin_res_h, fin_res_w)
                )
            else:
                rgb_hids[rgb_k] = f.interpolate(rgb_hid, (C, fin_res_h, fin_res_w))
            if d_hid.shape[-1] > fin_res_w:
                d_hids[d_k] = f.adaptive_avg_pool3d(d_hid, (C, fin_res_h, fin_res_w))
            else:
                d_hids[d_k] = f.interpolate(d_hid, (C, fin_res_h, fin_res_w))
            hidden_states[rgb_k] = rgb_hids[rgb_k] + d_hids[d_k]
        hidden_states = torch.stack([emb for emb in hidden_states.values()])
        return hidden_states


class AddPosEmb(nn.Module, ABC):
    """
    Add positional embedding to the original embeddings
    """
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def __call__(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Add positional embedding to original embeddings

        Args:
            hidden_states (torch.Tensor): original embeddings
        Return:
            torch.Tensor: output embeddings
        """
        ...


class AddPosEmb3D(AddPosEmb):
    def __call__(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Add positional embedding to original embeddings with [N x B x C x H x W] shape

        Args:
            hidden_states (torch.Tensor): original embeddings with [N x B x C x H x W] shape
        Return:
            torch.Tensor: output embeddings with the same shape
        """
        hidden_states = rearrange(hidden_states, "n b c h w -> b n h w c")
        _, n, h, w, c = hidden_states.shape
        pos_emb = get_pos_embed_nd(n, h, w, emb_dim=c).to(hidden_states.device)
        hidden_states = hidden_states + pos_emb
        hidden_states = rearrange(hidden_states, "b n h w c -> n b c h w")
        return hidden_states


class AddPosEmb4D(AddPosEmb):
    def __call__(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Add positional embedding to original embeddings with [N x B x M x C x H x W] shape

        Args:
            hidden_states (torch.Tensor): original embeddings with [N x B x M x C x H x W] shape
        Return:
            torch.Tensor: output embeddings with the same shape
        """
        hidden_states = rearrange(hidden_states, "n b m c h w -> b n m h w c")
        _, n, m, h, w, c = hidden_states.shape
        pos_emb = get_pos_embed_nd(n, m, h, w, emb_dim=c).to(hidden_states.device)
        hidden_states = hidden_states + pos_emb
        hidden_states = rearrange(hidden_states, "b n m h w c -> n b m c h w")
        return hidden_states
