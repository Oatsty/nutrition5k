from yacs.config import CfgNode as CN

from .base import BaseTrainer
from .train_mask import MaskTrainer
from .train_mask_lp import MaskLPTrainer
from .train_mask_lp_branch import MaskLPBranchTrainer
from .train_multi_mask import MultiMaskTrainer
from .train_noisy import NoisyTrainer
from .train_normal import NormalTrainer


def get_trainer(config: CN) -> BaseTrainer:
    trainer_name = config.TRAIN.TRAINER
    if trainer_name == "normal":
        return NormalTrainer(config)
    elif trainer_name == "noisy":
        return NoisyTrainer(config)
    elif trainer_name == "mask":
        return MaskTrainer(config)
    elif trainer_name == "mask_lp":
        return MaskLPTrainer(config)
    elif trainer_name == "multi_mask":
        return MultiMaskTrainer(config)
    elif trainer_name == "mask_lp_branch":
        return MaskLPBranchTrainer(config)
    else:
        raise ValueError(f"Invalid trainer name: {trainer_name}")
