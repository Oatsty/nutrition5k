from typing import Any, Callable

import torch
import torch.nn.functional as F
from dataset import Ingr, Metadata
from torch import nn
from yacs.config import CfgNode as CN

num_ingrs = 555


def get_keys(config: CN):
    loss = config.TRAIN.LOSS
    if loss == "multi":
        return ["cal", "mass", "fat", "carb", "protein"]
    elif loss == "multi_ingrs":
        return ["cal", "mass", "fat", "carb", "protein", "ingrs"]
    else:
        raise ValueError(f"Invalid loss function: {loss}")


def loss_func_multi(
    outputs: dict[str, torch.Tensor],
    metadata: list[Metadata],
    device: torch.device,
    **kwargs,
) -> dict[str, torch.Tensor]:
    loss_multi = {
        x: torch.tensor(0.0) for x in ["cal", "mass", "fat", "carb", "protein"]
    }
    for key in loss_multi.keys():
        target = torch.tensor([met.__getattribute__(key) for met in metadata]).to(
            device
        )
        loss_multi[key] = nn.L1Loss(**kwargs)(outputs[key].squeeze(), target)
    return loss_multi


def ingr_id_to_int(ingr_id: str) -> int:
    return int(ingr_id.split("_")[1])


def ingrs_to_tensor(ingrs: list[Ingr], device: torch.device) -> torch.Tensor:
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
    ingrs_list: list[list[Ingr]] = [met.__getattribute__("ingrs") for met in metadata]
    target_ingrs = torch.stack([ingrs_to_tensor(ingrs, device) for ingrs in ingrs_list])
    return target_ingrs


def loss_func_multi_ingr(
    outputs: dict[str, torch.Tensor],
    metadata: list[Metadata],
    device: torch.device,
    **kwargs,
) -> dict[str, torch.Tensor]:
    loss_multi = loss_func_multi(outputs, metadata, device, **kwargs)
    target_ingrs = met_to_ingr_tensor(metadata, device)
    loss_multi["ingrs"] = (
        nn.CrossEntropyLoss()(outputs["ingrs"], target_ingrs) / 6.31
    )  # ln(555)
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


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class AverageMeterDict(object):
    def __init__(self, keys):
        self.keys = keys
        self.average_meter_dict = {key: AverageMeter() for key in keys}

    def reset(self):
        for key in self.keys:
            self.average_meter_dict[key].reset()

    def update(self, val: dict[str, Any], n=1):
        for key in val.keys():
            if key not in self.keys:
                raise ValueError(f"{self} does not have key {key}")
            self.average_meter_dict[key].update(val[key], n)

    def get_sum(self, key):
        if key not in self.keys:
            raise ValueError(f"{self} does not have key {key}")
        return self.average_meter_dict[key].sum

    def get_avg(self, key):
        if key not in self.keys:
            raise ValueError(f"{self} does not have key {key}")
        return self.average_meter_dict[key].avg

    def iter_sum(self):
        for key in self.keys:
            yield key, self.average_meter_dict[key].sum

    def iter_avg(self):
        for key in self.keys:
            yield key, self.average_meter_dict[key].avg

    def get_keys(self):
        return self.average_meter_dict.keys()
