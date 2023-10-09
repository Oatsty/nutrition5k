from typing import Callable

import torch
import torch.nn.functional as F
from dataset import Ingr, Metadata
from torch import nn
from yacs.config import CfgNode as CN

num_ingrs = 555


def get_keys(config: CN):
    """
    Get model output keys. Depends on the loss function.

    Args:
        config (CN): config
    Return:
        keys (list[str]): list of keys
    """
    loss = config.TRAIN.LOSS
    if loss == "multi":
        return ["cal", "mass", "fat", "carb", "protein"]
    elif loss == "multi_ingrs":
        return ["cal", "mass", "fat", "carb", "protein", "ingrs"]
    else:
        raise ValueError(f"Invalid loss function: {loss}")


class Lploss(nn.Module):
    """
    Lp Loss Module

    Args:
        p (float): p value [0,1]
        reduction (str): "mean" or "sum"
    """
    def __init__(self, p: float, reduction: str = "mean") -> None:
        super(Lploss, self).__init__()
        self.reduction = reduction
        self.p = p

    def forward(self, output: torch.Tensor, target: torch.Tensor):
        """
        Calculate Lp loss

        Args:
            output (torch.Tensor): Model outputs for a specific nutrient. Desired shape [N]
            target (torch.Tensor): Target values. Desired shape [N]
        Return:
            loss (torch.Tensor): Scalar tensor for Lp loss
        """
        N = output.shape[0]
        l1loss = (output - target).abs()
        entropy = torch.clip(self.p * torch.log(l1loss + 1e-20), -1e20, 20)
        loss = torch.exp(entropy)
        loss = loss.sum()
        loss = torch.clip(loss, 0, 1e20)
        loss = torch.exp(torch.log(loss + 1e-20) / self.p)
        if self.reduction == "mean":
            loss = loss / N
        return loss


def loss_func_multi(
    outputs: dict[str, torch.Tensor],
    metadata: list[Metadata],
    device: torch.device,
    p: float = 1.0,
    **kwargs,
) -> dict[str, torch.Tensor]:
    """
    Calculate loss for all nutritional values (cal, mass, fat, carb, protein)

    Args:
        outputs (dict[str, torch.Tensor]): Model outputs with keys = [cal, mass, fat, carb, protein]
        metadata list[Metadata]: Target values represent by a list of metadata.
        device (torch.device): Device
        p (float): Lp loss p [0,1]
    Return:
        loss_multi (dict[str, torch.Tensor]): Dictionary of losses for all nutrients. Same keys as model outputs.
    """
    loss_multi = {
        x: torch.tensor(0.0) for x in ["cal", "mass", "fat", "carb", "protein"]
    }
    for key in loss_multi.keys():
        target = torch.tensor([met.__getattribute__(key) for met in metadata]).to(
            device
        )
        if p == 1.0:
            loss_multi[key] = nn.L1Loss(**kwargs)(outputs[key].squeeze(-1), target)
        else:
            loss_multi[key] = Lploss(p, **kwargs)(outputs[key].squeeze(-1), target)
    return loss_multi


def ingr_id_to_int(ingr_id: str) -> int:
    return int(ingr_id.split("_")[1])


def ingrs_to_tensor(ingrs: list[Ingr], device: torch.device) -> torch.Tensor:
    """
    Generate a Tensor from ingredient list. Ingredient that are not included in the dish will be zero.
    Ingredients in the dish will be weight according to its weight and sum to 1.

    Args:
        ingrs (list[Ingr]): List of ingredients
        device (torch.device): Device
    Return:
        torch.Tensor: A 1d tensor representing the ingredient list
    """
    ingr_ids = torch.tensor([ingr_id_to_int(ingr.id) for ingr in ingrs]).to(device)
    ingr_grams = (
        torch.tensor([float(ingr.grams) for ingr in ingrs]).to(device).unsqueeze(1)
    )
    ingrs_one_hot: torch.Tensor = F.one_hot(ingr_ids, num_classes=num_ingrs)
    ingrs_target = ingrs_one_hot * ingr_grams
    ingrs_target = ingrs_target.sum(0)
    ingrs_target = ingrs_target / ingrs_target.sum()
    return ingrs_target


def met_to_ingr_tensor(metadata: list[Metadata], device: torch.device) -> torch.Tensor:
    """
    Convert a list of ingredient lists to a Tensor

    Args:
        metadata (list[Metadata]): Metadatas of all images in the batch
        device (torch.device): Device
    Return:
        torch.Tensor: A 2d tensor representing all the ingredient lists
    """
    ingrs_list: list[list[Ingr]] = [met.__getattribute__("ingrs") for met in metadata]
    target_ingrs = torch.stack([ingrs_to_tensor(ingrs, device) for ingrs in ingrs_list])
    return target_ingrs


def loss_func_multi_ingr(
    outputs: dict[str, torch.Tensor],
    metadata: list[Metadata],
    device: torch.device,
    **kwargs,
) -> dict[str, torch.Tensor]:
    """
    Calculate loss for all nutritional values (cal, mass, fat, carb, protein) and ingredients

    Args:
        outputs (dict[str, torch.Tensor]): Model outputs with keys = [cal, mass, fat, carb, protein, ingrs]
        metadata list[Metadata]: Target values represent by a list of metadata.
        device (torch.device): Device
        p (float): Lp loss p [0,1]
    Return:
        loss_multi (dict[str, torch.Tensor]): Dictionary of losses for all nutrients and ingredients. Same keys as model outputs.
    """
    loss_multi = loss_func_multi(outputs, metadata, device, **kwargs)
    target_ingrs = met_to_ingr_tensor(metadata, device)
    loss_multi["ingrs"] = (
        nn.CrossEntropyLoss()(outputs["ingrs"], target_ingrs) / 6.31  # ln(555)
    )
    return loss_multi


def get_loss(config: CN) -> Callable[..., dict[str, torch.Tensor]]:
    loss = config.TRAIN.LOSS
    if loss == "multi":
        return loss_func_multi
    elif loss == "multi_ingrs":
        return loss_func_multi_ingr
    else:
        raise ValueError(f"Invalid loss function: {loss}")


def get_candidate(
    outputs: dict[str, torch.Tensor],
    metadata: list[Metadata],
    loss_multi: dict[str, torch.Tensor],
    k: int = 3,
    weight: float = 0.1,
) -> dict[str, tuple[Metadata, float]]:
    """
    Update metadatas of dishes with high loss. High losses might resulted from noisy labels

    Args:
        outputs (dict[str, torch.Tensor]): Model outputs with keys = [cal, mass, fat, carb, protein, ingrs] or [cal, mass, fat, carb, protein]
        metadata (list[Metadata]): list of target metadata
        loss_multi (dict[str, torch.Tensor]): losses
        k (int): number of metadatas to possibly update (default=3)
        weight (float): update weight [0,1]. (default=0.1)
    Return:
        dict[str, (Metadata, float)]: dictionary of metadata to be updated and its loss
    """
    outputs_cands: dict[str, tuple[Metadata, float]] = {}
    individual_losses = torch.stack(list(loss_multi.values())).sum(0)
    assert len(individual_losses) == len(outputs["cal"])
    cand_ids = individual_losses.argsort(descending=True)[:k]
    for id in cand_ids:
        dish_id = metadata[id].dish_id
        ingrs = metadata[id].ingrs
        loss = individual_losses[id].item()
        new_metadata_dict = {}
        for key, pred_val in outputs.items():
            new_metadata_dict[key] = pred_val * weight + metadata[id].__getattribute__(
                key
            ) * (1 - weight)
        outputs_cands[dish_id] = (
            Metadata(
                dish_id=dish_id,
                ingrs=ingrs,
                **{k: v[id].item() for k, v in outputs.items()},
            ),
            loss,
        )
    return outputs_cands
