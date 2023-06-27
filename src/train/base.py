from abc import ABC, abstractclassmethod, abstractstaticmethod
from yacs.config import CfgNode as CN

import torch

from model import BaseModel


class BaseTrainer(ABC):
    @abstractstaticmethod
    def train(config: CN, model: BaseModel, device: torch.device) -> None:
        ...