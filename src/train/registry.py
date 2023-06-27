from yacs.config import CfgNode as CN

from .base import BaseTrainer
from .train_mask import MaskTrainer
from .train_noisy import NoisyTrainer
from .train_normal import NormalTrainer

def get_trainer(config: CN) -> BaseTrainer:
    trainer_name = config.TRAIN.TRAINER
    if trainer_name == 'normal':
        return NormalTrainer()
    elif trainer_name == 'noisy':
        return NoisyTrainer()
    elif trainer_name == 'mask':
        return MaskTrainer()
    else:
        raise ValueError(f'Invalid trainer name: {trainer_name}')