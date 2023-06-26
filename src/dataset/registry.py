from yacs.config import CfgNode as CN

from .base_dataset import BaseDataset
from .nutrition5k_dataset import make_dataset as make_nutrition5k_dataset
from .recipe1m_dataset import make_dataset as make_recipe1m_dataset

def make_dataset(config: CN,**kwargs) -> dict[str, BaseDataset]:
    dataset_name = config.DATA.NAME
    if dataset_name == 'nutrition5k':
        return make_nutrition5k_dataset(config,**kwargs)
    elif dataset_name == 'recipe1m':
        return make_recipe1m_dataset(config, **kwargs)
    else:
        raise ValueError(f'Unkown dataset: {dataset_name}')