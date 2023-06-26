from typing import Callable
import torch
from torch import nn
import torch.nn.functional as F
from yacs.config import CfgNode as CN

from dataset.nutrition5k_dataset import Ingr, Metadata

num_ingrs = 555

def loss_func_multi(outputs: dict[str,torch.Tensor], metadata: list[Metadata], criterion: nn.Module, device: torch.device) -> dict[str, torch.Tensor]:
    loss_multi = {x: torch.tensor(0.) for x in ['cal','mass','fat','carb','protein']}
    for key in loss_multi.keys():
        target = torch.tensor([met.__getattribute__(key) for met in metadata]).to(device)
        loss_multi[key] = criterion(outputs[key].squeeze(), target)
    return loss_multi

def ingr_id_to_int(ingr_id: str) -> int:
    return int(ingr_id.split('_')[1])

def ingrs_to_tensor(ingrs: list[Ingr], device: torch.device) -> torch.Tensor:
    ingr_ids = torch.tensor([ingr_id_to_int(ingr.id) for ingr in ingrs]).to(device)
    ingr_grams = torch.tensor([float(ingr.grams) for ingr in ingrs]).to(device).unsqueeze(1)
    ingrs_one_hot: torch.Tensor = F.one_hot(ingr_ids, num_classes=num_ingrs)
    ingrs_target = ingrs_one_hot * ingr_grams
    ingrs_target = ingrs_target.sum(0)
    ingrs_target = ingrs_target / ingrs_target.sum()
    return ingrs_target

def met_to_ingr_tensor(metadata: list[Metadata], device: torch.device) -> torch.Tensor:
    ingrs_list: list[list[Ingr]] = [met.__getattribute__('ingrs') for met in metadata]
    target_ingrs = torch.stack([ingrs_to_tensor(ingrs,device) for ingrs in ingrs_list])
    return target_ingrs

def loss_func_multi_ingr(outputs: dict[str,torch.Tensor], metadata: list[Metadata], criterion: nn.Module, device: torch.device) -> dict[str, torch.Tensor]:
    loss_multi = loss_func_multi(outputs,metadata,criterion,device)
    target_ingrs = met_to_ingr_tensor(metadata,device)
    loss_multi['ingrs'] = nn.CrossEntropyLoss()(outputs['ingrs'],target_ingrs) / 6.31 # ln(555)
    return loss_multi

def get_loss(config: CN) -> Callable:
    loss = config.TRAIN.LOSS
    if loss == 'multi':
        return loss_func_multi
    elif loss == 'multi_ingrs':
        return loss_func_multi_ingr
    else:
        exit(1)