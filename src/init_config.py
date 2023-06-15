import argparse
import logging
import random
import os
import yaml
from yacs.config import CfgNode as CN
from pathlib import Path
from typing import Tuple

import numpy as np
import torch

_C = CN()

# -----------------------------------------------------------------------------
# Data settings
# -----------------------------------------------------------------------------
_C.DATA = CN()
_C.DATA.IMGS_DIR = '/srv/datasets2/nutrition5k_dataset/imagery/realsense_overhead'
_C.DATA.METADATAS_PATH = '/srv/datasets2/nutrition5k_dataset/metadata/dish_metadata_cafe1.csv'
_C.DATA.SPLITS_TRAIN_PATH = '/srv/datasets2/nutrition5k_dataset/dish_ids/splits/depth_train_ids.txt'
_C.DATA.SPLITS_TEST_PATH = '/srv/datasets2/nutrition5k_dataset/dish_ids/splits/depth_test_ids.txt'

# -----------------------------------------------------------------------------
# model
# -----------------------------------------------------------------------------
_C.MODEL = CN()
_C.MODEL.NAME = 'inceptionv2'
_C.MODEL.PRETRAINED = 'inception_resnet_v2'

# -----------------------------------------------------------------------------
# train
# -----------------------------------------------------------------------------
_C.TRAIN = CN()
_C.TRAIN.BATCH_SIZE = 128
_C.TRAIN.LR = 0.01
_C.TRAIN.NUM_EPOCHS = 20
_C.TRAIN.WEIGHT_DECAY = 1e-4
_C.TRAIN.CKPT = None
_C.TRAIN.FINETUNE = False
_C.TRAIN.LAYERS = 1
_C.TRAIN.SEED = 12345

# -----------------------------------------------------------------------------
# eval
# -----------------------------------------------------------------------------
_C.EVAL = CN()

# -----------------------------------------------------------------------------
# misc
# -----------------------------------------------------------------------------
_C.SAVE_PATH = 'models/simpleinceptionv2.pt'
_C.TITLE = ['simple inceptionv2 test']


def _update_config_from_file(config, cfg_file):
    config.defrost()
    with open(cfg_file, 'r') as f:
        yaml_cfg = yaml.load(f, Loader=yaml.FullLoader)

    for cfg in yaml_cfg.setdefault('BASE', ['']):
        if cfg:
            _update_config_from_file(
                config, os.path.join(os.path.dirname(cfg_file), cfg)
            )
    print('=> merge config from {}'.format(cfg_file))
    config.merge_from_file(cfg_file)
    config.freeze()

def update_config(config, args):
    def _check_args(name):
        if hasattr(args, name) and eval(f'args.{name}'):
            return True
        return False
    
    if _check_args('cfg'):
        _update_config_from_file(config, args.cfg)

    config.defrost()

    # merge from specific arguments
    if _check_args('imgs_dir'):
        config.DATA.IMGS_DIR = args.imgs_dir
    if _check_args('metadatas_path'):
        config.DATA.METADATAS_PATH = args.metadatas_path
    if _check_args('splits_train_path'):
        config.DATA.SPLITS_TRAIN_PATH = args.splits_train_path
    if _check_args('splits_test_path'):
        config.DATA.SPLITS_TEST_PATH = args.splits_test_path
    if _check_args('model_name'):
        config.MODEL.NAME = args.model_name
    if _check_args('pretrained_model'):
        config.MODEL.PRETRAINED = args.pretrained_model
    if _check_args('batch_size'):
        config.TRAIN.BATCH_SIZE = args.batch_size
    if _check_args('lr'):
        config.TRAIN.LR = args.lr
    if _check_args('num_epochs'):
        config.TRAIN.NUM_EPOCHS = args.num_epochs
    if _check_args('weight_decay'):
        config.TRAIN.WEIGHT_DECAY = args.weight_decay
    if _check_args('ckpt'):
        config.TRAIN.CKPT = args.ckpt
    if _check_args('finetune'):
        config.TRAIN.FINETUNE = True
    if _check_args('layers'):
        config.TRAIN.LAYERS = args.layers
    if _check_args('save_path'):
        config.SAVE_PATH = args.save_path
    if _check_args('title'):
        config.TITLE = args.title

    config.freeze()

def get_config(args):
    """Get a yacs CfgNode object with default values."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    config = _C.clone()
    update_config(config, args)
    return config

def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def init_logger(log_dir: str, log_file: str) -> logging.Logger:
    logger = logging.getLogger()
    format_str = r'[%(asctime)s] %(message)s'
    logging.basicConfig(
        level=logging.INFO,
        datefmt=r'%Y/%m/%d %H:%M:%S',
        format=format_str
    )
    log_dir_path = Path(log_dir)
    log_dir_path.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(str(log_dir_path / log_file))
    fh.setFormatter(logging.Formatter(format_str))
    logger.addHandler(fh)
    return logger

def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument('--imgs-dir', type=str)
    parser.add_argument('--metadatas-path', type=str)
    parser.add_argument('--splits-train-path', type=str)
    parser.add_argument('--splits-test-path', type=str)
    parser.add_argument('--model-name', type=str)
    parser.add_argument('--pretrained-model', type=str)
    parser.add_argument('--batch-size', type=int)
    parser.add_argument('--lr', type=float)
    parser.add_argument('--num-epochs', type=int)
    parser.add_argument('--weight-decay', type=float)
    parser.add_argument('--ckpt', type=str)
    parser.add_argument('--finetune', action='store_true')
    parser.add_argument('--layers', type=int)
    parser.add_argument('--thresh', type=float)
    parser.add_argument('--save-path', type=str)
    parser.add_argument('--cfg', type=str)
    parser.add_argument('--title', type=str, nargs='+')
    
    return parser

def get_arguments() -> Tuple[argparse.Namespace,CN]:
    parser = get_parser()
    args = parser.parse_args()
    config = get_config(args)
    return args, config

if __name__ == '__main__':
    _, config = get_arguments()
    logger = init_logger('.', 'log.txt')
    set_random_seed(config.TRAIN.SEED)
    dump_path = os.path.join('config',os.path.splitext(config.SAVE_PATH)[0] + '.yaml')
    os.makedirs(os.path.dirname(dump_path), exist_ok=True)
    with open(dump_path,'w') as f:
        f.write(config.dump())
    logger.info(f'Config Path: {dump_path}')