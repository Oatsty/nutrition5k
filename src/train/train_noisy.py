import logging
import random
from tqdm import tqdm
from yacs.config import CfgNode as CN

import torch
import torch.optim as optim
from torchvision import transforms

from .base import BaseTrainer
from custom_utils import get_candidate
from dataset import Metadata
from model import BaseModel

logger = logging.getLogger()


class NoisyTrainer(BaseTrainer):
    def init_train(self, config: CN, model: BaseModel) -> None:
        self.warmup_epochs = config.TRAIN.WARMUP_EPOCHS
        self.noise_removal_epochs = config.TRAIN.NOISE_REMOVAL_EPOCHS
        self.remove_rate = config.TRAIN.REMOVE_RATE
        self.optimizer = optim.Adam(
            model.parameters(),
            lr=config.TRAIN.LR,
            weight_decay=config.TRAIN.WEIGHT_DECAY,
        )
        self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.95)

    def train_one_epoch(
        self, model: BaseModel, epoch: int, phase: str, device: torch.device
    ) -> None:
        outputs_cands = {}
        if epoch < self.warmup_epochs:
            noisy_phase = "warmup"
        elif epoch < self.warmup_epochs + self.noise_removal_epochs:
            noisy_phase = "noise_removal"
        else:
            noisy_phase = "normal"
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
            if noisy_phase == "noise_removal" or noisy_phase == "warmup":
                outputs = model(rgb_img, depth_img, skip_attn=True, depth=2)
            else:
                outputs = model(rgb_img, depth_img)
            if phase == "train" and noisy_phase == "noise_removal":
                loss_multi = self.loss_func(outputs, metadata, device, reduction="none")
                cands = get_candidate(
                    outputs,
                    metadata,
                    loss_multi,
                    k=int(len(rgb_img) * self.remove_rate * 2),
                )
                outputs_cands.update(cands)
                loss_multi = {k: v.mean() for k, v in loss_multi.items()}
            else:
                loss_multi = self.loss_func(outputs, metadata, device)

            loss = sum(loss_multi.values())
            assert isinstance(loss, torch.Tensor)
            if phase == "train":
                loss.backward()
                self.optimizer.step()
            self.avg_meter.update({"total": loss.item()}, len(rgb_img))
            self.avg_meter.update(loss_multi, len(rgb_img))
