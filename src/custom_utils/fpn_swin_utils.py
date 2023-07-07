from abc import ABC, abstractmethod

import torch
import torch.nn.functional as f
from einops import rearrange

from .position_embedding import get_pos_embed_nd


class Mask(ABC):
    def __init__(self, mask_weight: float) -> None:
        super().__init__()
        self.mask_weight = mask_weight

    @abstractmethod
    def __call__(
        self,
        feats_hid: dict[str, torch.Tensor] | dict[str, torch.FloatTensor],
        mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        ...


class DefaultMask(Mask):
    def __call__(
        self,
        feats_hid: dict[str, torch.Tensor] | dict[str, torch.FloatTensor],
        mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
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
    def __init__(self, mask_weight: float, mask_dim: int) -> None:
        super().__init__(mask_weight)
        self.mask_dim = mask_dim

    def __call__(
        self,
        feats_hid: dict[str, torch.FloatTensor] | dict[str, torch.Tensor],
        mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
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
    def __init__(self, mask_weight: float, mask_dim: int) -> None:
        super().__init__(mask_weight)
        self.mask_dim = mask_dim

    def __call__(
        self,
        feats_hid: dict[str, torch.FloatTensor] | dict[str, torch.Tensor],
        mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
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


class Resize(ABC):
    def __init__(self, resolution_level: str) -> None:
        super().__init__()
        self.resolution_level = resolution_level

    @abstractmethod
    def __call__(
        self,
        rgb_hids: dict[str, torch.Tensor],
        d_hids: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        ...


class Resize2D(Resize):
    def __call__(
        self,
        rgb_hids: dict[str, torch.Tensor],
        d_hids: dict[str, torch.Tensor],
    ) -> torch.Tensor:
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


class AddPosEmb(ABC):
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def __call__(self, hidden_states: torch.Tensor) -> torch.Tensor:
        ...


class AddPosEmb3D(AddPosEmb):
    def __call__(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = rearrange(hidden_states, "n b c h w -> b n h w c")
        _, n, h, w, c = hidden_states.shape
        pos_emb = get_pos_embed_nd(n, h, w, emb_dim=c).to(hidden_states.device)
        hidden_states = hidden_states + pos_emb
        hidden_states = rearrange(hidden_states, "b n h w c -> n b c h w")
        return hidden_states


class AddPosEmb4D(AddPosEmb):
    def __call__(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = rearrange(hidden_states, "n b m c h w -> b n m h w c")
        _, n, m, h, w, c = hidden_states.shape
        pos_emb = get_pos_embed_nd(n, m, h, w, emb_dim=c).to(hidden_states.device)
        hidden_states = hidden_states + pos_emb
        hidden_states = rearrange(hidden_states, "b n m h w c -> n b m c h w")
        return hidden_states
