import os
import sys

pth = "/".join(sys.path[0].split("/")[:-1])
openseed_pth = "/".join([pth, "OpenSeeD"])
sys.path.insert(0, openseed_pth)
sys.path.append(pth)

import logging

import init_config
import torch
from model import get_model
from train import get_trainer

logger = logging.getLogger()


def main():
    # init configurations from config file
    _, config = init_config.get_arguments()
    os.makedirs(os.path.dirname(config.SAVE_PATH), exist_ok=True)

    # set random seed
    init_config.set_random_seed(config.TRAIN.SEED)

    # init model
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = get_model(config, device)
    model.to(device)
    if config.TRAIN.CKPT:
        model.load_state_dict(torch.load(config.TRAIN.CKPT))

    # init logger
    log_path = os.path.join("log", os.path.splitext(config.SAVE_PATH)[0] + ".txt")
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    logger = init_config.init_logger(
        os.path.dirname(log_path), os.path.basename(log_path)
    )
    logger.info(config.dump())

    # dump current config
    dump_path = os.path.join("config", os.path.splitext(config.SAVE_PATH)[0] + ".yaml")
    os.makedirs(os.path.dirname(dump_path), exist_ok=True)
    with open(dump_path, "w") as f:
        f.write(config.dump())  # type: ignore

    # train
    trainer = get_trainer(config)
    trainer.train(config, model, device=device)


if __name__ == "__main__":
    main()
