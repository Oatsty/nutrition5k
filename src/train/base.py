import logging
from abc import ABC, abstractmethod

import torch
import torch.optim as optim
from custom_utils import AverageMeter, AverageMeterDict, get_keys, get_loss
from dataset import collate_fn, make_dataset
from model import BaseModel
from timm.scheduler import CosineLRScheduler
from torch.utils.data import DataLoader
from yacs.config import CfgNode as CN

logger = logging.getLogger()


class BaseTrainer(ABC):
    """
    Abstract class for trainers

    Args:
        config (CN): config
    """
    def __init__(self, config: CN) -> None:
        super().__init__()
        self.num_epochs = config.TRAIN.NUM_EPOCHS
        self.batch_size = config.TRAIN.BATCH_SIZE
        if config.MODEL.PRETRAINED == "microsoft/swin-tiny-patch4-window7-224":
            self.resize_sizes = [
                (256, 352),
                (288, 384),
                (320, 448),
                (352, 480),
                (384, 512),
                (480, 640),
            ]
        else:
            self.resize_sizes = [(384, 512)]
        self.loss_func = get_loss(config)
        self.dataset = make_dataset(config)
        self.dataloader = {
            x: DataLoader(
                self.dataset[x],
                batch_size=self.batch_size,
                num_workers=4,
                shuffle=True if x == "train" else False,
                collate_fn=collate_fn,
            )
            for x in ["train", "test"]
        }
        keys = get_keys(config)
        keys.insert(0, "total")
        self.avg_meter = AverageMeterDict(keys)
        self.mean = self.dataset["train"].mean_metadata
        self.std = self.dataset["train"].std_metadata

    @abstractmethod
    def init_train(self, config: CN, model: BaseModel) -> None:
        """
        Init optimizer and scheduler

        Args:
            config (CN): config
            model (BaseModel): model in this training
        """
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

    @abstractmethod
    def train_one_epoch(
        self, model: BaseModel, epoch: int, phase: str, device: torch.device
    ) -> None:
        """
        Train for one epoch

        Args:
            model (BaseModel): model in training
            epoch (int): current epoch
            phase (str): current phase, train/test etc.
            device (torch.device): device
        """
        ...

    def display_loss(self, phase: str):
        """
        Log MAE and PMAE for all nutrients and the average PMAE

        Args:
            phase (str): current phase, train/test etc.
        """
        apmae = AverageMeter()
        for key, key_loss in self.avg_meter.iter_avg():
            if key == "total":
                logger.info(f"{phase} {key} loss: {key_loss:.4f}")
                continue
            if key == "ingrs":
                logger.info(f"{phase} {key} percent loss: {key_loss:.4f}")
                continue
            mean = self.mean.__getattribute__(key)
            std = self.std.__getattribute__(key)
            logger.info(f"{phase} {key} loss: {key_loss * std:.4f}")
            logger.info(f"{phase} {key} percent loss: {key_loss * std / mean:.4f}")
            apmae.update(key_loss * std / mean)
        logger.info(f"{phase} average percent loss: {apmae.avg:.4f}")

    def train(self, config: CN, model: BaseModel, device: torch.device) -> None:
        """
        Train the model for num_epochs and log the loss for every epoch

        Args:
            config (CN): config
            model (BaseModel): model in training
            device (torch.device): device
        """
        self.init_train(config, model)
        for epoch in range(self.num_epochs):
            print(f'lr = {self.optimizer.param_groups[0]["lr"]}')
            logger.info(f"Epoch {epoch+1}/{self.num_epochs}")
            for phase in ["train", "test"]:
                print(phase)
                self.avg_meter.reset()
                if phase == "train":
                    model.train()
                if phase == "test":
                    model.eval()
                with torch.set_grad_enabled(phase == "train"):
                    self.train_one_epoch(model, epoch, phase, device)
                self.display_loss(phase)
            self.scheduler.step(epoch + 1)

        torch.save(model.state_dict(), config.SAVE_PATH)
