import os
import sys

sys.path.insert(0, "/home/parinayok/nutrition5k/OpenSeeD")
sys.path.append("/home/parinayok/nutrition5k")

import logging
from typing import Callable

import init_config
import torch
import torchvision
from custom_utils import get_loss
from dataset import Metadata, collate_fn, make_dataset
from model import get_model
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from yacs.config import CfgNode as CN

logger = logging.getLogger()


def train(
    config: CN,
    model: nn.Module,
    loss_func: Callable,
    criterion: nn.Module,
    device: torch.device,
):
    print(" ".join(config.TITLE))
    batch_size = config.TRAIN.BATCH_SIZE
    resize_size = (config.EVAL.HEIGHT, config.EVAL.WIDTH)
    # データセットの準備
    dataset = make_dataset(config)
    dataloader = DataLoader(
        dataset["test"],
        batch_size=batch_size,
        num_workers=8,
        shuffle=False,
        collate_fn=collate_fn,
    )
    running_loss = 0.0
    running_loss_multi = {}
    with torch.no_grad():
        for batch in tqdm(dataloader):
            rgb_img = batch["rgb_img"]
            depth_img = batch["depth_img"]
            metadata: list[Metadata] = batch["metadata"]
            rgb_img = rgb_img.to(device)
            depth_img = depth_img.to(device)
            resize = transforms.Resize(resize_size)
            rgb_img = resize(rgb_img)
            depth_img = resize(depth_img)
            outputs = model(rgb_img, depth_img)
            loss_multi = loss_func(outputs, metadata, criterion, device)
            loss = sum(loss_multi.values())
            assert isinstance(loss, torch.Tensor)
            running_loss += loss.item() * len(rgb_img)
            for key in loss_multi.keys():
                if key in running_loss_multi.keys():
                    running_loss_multi[key] += loss_multi[key].item() * len(rgb_img)
                else:
                    running_loss_multi[key] = loss_multi[key].item() * len(rgb_img)

    running_loss /= len(dataset["test"])
    logger.info(f"loss: {running_loss:.4f}")
    for key in running_loss_multi.keys():
        running_loss_multi[key] /= len(dataset["test"])
        if key == "ingrs":
            logger.info(f"{key} percent loss: {running_loss_multi[key]:.4f}")
            continue
        mean = dataset["test"].mean_metadata.__getattribute__(key)
        std = dataset["test"].std_metadata.__getattribute__(key)
        logger.info(f"{key} loss: {running_loss_multi[key] * std:.4f}")
        logger.info(f"{key} percent loss: {running_loss_multi[key] * std / mean:.4f}")


def main():
    _, config = init_config.get_arguments()
    os.makedirs(os.path.dirname(config.SAVE_PATH), exist_ok=True)

    # モデルの準備
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = get_model(config, device)
    model.to(device)
    model.eval()
    model.load_state_dict(torch.load(config.SAVE_PATH))
    # 損失関数と最適化関数の定義
    loss_func = get_loss(config)
    criterion = nn.L1Loss()

    log_path = os.path.join("log", os.path.splitext(config.SAVE_PATH)[0] + "_eval.txt")
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    init_config.init_logger(os.path.dirname(log_path), os.path.basename(log_path))
    init_config.set_random_seed(config.TRAIN.SEED)
    logger.info(config.dump())

    train(config, model, loss_func, criterion, device=device)


if __name__ == "__main__":
    main()
