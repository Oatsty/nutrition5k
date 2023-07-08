import logging
import random

import torch
import torch.optim as optim
from dataset import Metadata
from model import BaseModel
from timm.scheduler import CosineLRScheduler
from torchvision import transforms
from tqdm import tqdm
from yacs.config import CfgNode as CN

from .base import BaseTrainer

logger = logging.getLogger()


class NormalTrainer(BaseTrainer):
    def init_train(self, config: CN, model: BaseModel) -> None:
        scheduler_warmup = config.TRAIN.SCHEDULER_WARMUP
        self.optimizer = optim.Adam(
            model.parameters(),
            lr=config.TRAIN.LR,
            weight_decay=config.TRAIN.WEIGHT_DECAY,
        )
        self.scheduler = CosineLRScheduler(
            self.optimizer,
            t_initial=config.TRAIN.NUM_EPOCHS - scheduler_warmup,
            lr_min=1e-7,
            warmup_t=scheduler_warmup,
            warmup_prefix=True,
        )

    def train_one_epoch(
        self, model: BaseModel, epoch: int, phase: str, device: torch.device
    ) -> None:
        for batch in tqdm(self.dataloader[phase]):
            rgb_img = batch["rgb_img"]
            depth_img = batch["depth_img"]
            metadata: list[Metadata] = batch["metadata"]
            rgb_img = rgb_img.to(device)
            depth_img = depth_img.to(device)
            if phase == "train":
                resize = transforms.Resize(random.choice(self.resize_sizes))
            else:
                resize = transforms.Resize(self.resize_sizes[-1])
            rgb_img = resize(rgb_img)
            depth_img = resize(depth_img)

            self.optimizer.zero_grad()
            outputs = model(rgb_img, depth_img)
            loss_multi = self.loss_func(outputs, metadata, device)
            loss = sum(loss_multi.values())
            assert isinstance(loss, torch.Tensor)
            if phase == "train":
                loss.backward()
                self.optimizer.step()
            self.avg_meter.update({"total": loss.item()}, len(rgb_img))
            loss_multi_item = {k: v.item() for k, v in loss_multi.items()}
            self.avg_meter.update(loss_multi_item, len(rgb_img))
