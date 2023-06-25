from dataclasses import dataclass, field
from time import process_time_ns
import numpy as np
import os
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader, default_collate
from torchvision import transforms
import torchvision.transforms.functional as TF
from typing import Any, Optional
from PIL import Image
from yacs.config import CfgNode as CN

@dataclass
class Ingr:
    id: str
    name: str
    grams: float
    cal: float
    fat: float
    carb: float
    protein: float

    def __str__(self) -> str:
        ingr_dict = {
            'id': self.id,
            'name': self.name,
            'grams': self.grams,
            'cal': self.cal,
            'fat': self.fat,
            'carb': self.carb,
            'protein': self.protein
        }
        return str(ingr_dict)


@dataclass
class Metadata:
    dish_id: str = field(default_factory=str)
    cal: float = field(default_factory=float)
    mass: float = field(default_factory=float)
    fat: float = field(default_factory=float) 
    carb: float = field(default_factory=float)
    protein: float = field(default_factory=float)
    ingrs: list[Ingr] = field(default_factory=list)

    def __str__(self) -> str:
        metadata_dict = {
            'dish_id': self.dish_id,
            'cal': self.cal,
            'mass': self.mass,
            'fat': self.fat,
            'carb': self.carb,
            'protein': self.protein,
            'ingrs': self.ingrs
        }
        return str(metadata_dict)

class Nutrition5kDataset(Dataset):
    def __init__(self, imgs_dir: Path, metadatas_path: Path, splits: list[str], transform: Optional[transforms.Compose] = None) -> None:
        super(Nutrition5kDataset, self).__init__()
        self.imgs_dir = imgs_dir
        self.metadatas_path = metadatas_path
        self.splits = splits
        self.metadatas_dict: dict[str,Metadata] = {}
        self.mean_metadata = Metadata('mean', 255.,218.,12.7,19.3,18.1)
        self.std_metadata = Metadata('std', 221.,163.,13.4,22.3,20.2)
        if isinstance(transform,transforms.Compose):
            self.transform = transform
        else:
            self.transform = transforms.Compose([
                # transforms.Resize(256),
                # transforms.CenterCrop((256,256)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])
        self.transform_depth = transforms.Compose([
            # transforms.Resize(256),
            # transforms.CenterCrop((256,256)),
            transforms.PILToTensor(),
        ])
        self.init_metadatas()

    @staticmethod
    def transform_(rgb_img,depth_img):
        # rotate
        params = transforms.RandomRotation.get_params([-180,180])
        rgb_img = TF.rotate(rgb_img,params)
        depth_img = TF.rotate(depth_img,params,fill=depth_img.max().item())

        if np.random.rand() < 0.5:
            rgb_img = TF.hflip(rgb_img)
            depth_img = TF.hflip(depth_img)
        return rgb_img, depth_img

    def init_metadatas(self):
        mean_metadata = self.mean_metadata
        std_metadata = self.std_metadata
        for line in open(self.metadatas_path,'r').readlines():
            line = line.rstrip()
            data_list = line.split(',')
            data_dict = Metadata()
            dish_id = data_list[0]
            data_dict.dish_id = dish_id
            data_dict.cal = (float(data_list[1]) - mean_metadata.cal) / std_metadata.cal
            data_dict.mass = (float(data_list[2]) - mean_metadata.mass) / std_metadata.mass
            data_dict.fat = (float(data_list[3]) - mean_metadata.fat) / std_metadata.fat
            data_dict.carb = (float(data_list[4]) - mean_metadata.carb) / std_metadata.carb
            data_dict.protein = (float(data_list[5]) - mean_metadata.protein) / std_metadata.protein
            data_dict.ingrs = []
            for id, name, grams, cal, fat, carb, protein in zip(*[data_list[x::7] for x in range(6,13)]): #type: ignore
                ingr = Ingr(**{
                    'id': id,
                    'name': name,
                    'grams': grams,
                    'cal': cal,
                    'fat': fat,
                    'carb': carb,
                    'protein': protein
                })
                data_dict.ingrs.append(ingr)
            assert len(data_list) == len(data_dict.ingrs) * 7 + 6
            self.metadatas_dict[dish_id] = data_dict

    def __len__(self):
        return len(self.splits)

    def __getitem__(self, index: int) -> dict[str,Any]:
        img_dir = Path.joinpath(self.imgs_dir,self.splits[index])
        rgb_path = Path.joinpath(img_dir,'rgb.png')
        depth_path = Path.joinpath(img_dir,'depth_raw.png')
        rgb_img = Image.open(rgb_path)
        depth_img = Image.open(depth_path)
        rgb_img = self.transform(rgb_img) 
        depth_img = self.transform_depth(depth_img).float() #type: ignore
        depth_img = transforms.Normalize(3091,1307)(depth_img)
        rgb_img, depth_img = self.transform_(rgb_img, depth_img)
        metadata = self.metadatas_dict[self.splits[index]]
        sample = {'rgb_img': rgb_img, 'depth_img': depth_img, 'metadata': metadata, 'rgb_path': str(rgb_path), 'depth_path': str(depth_path)}
        return sample

def make_dataset(config: Optional[CN], imgs_dir: str = '.', metadatas_path: str = '.', splits_train_path: str = '.', splits_test_path: str = '.', unnormalized_int_tensor: bool = False):
    if isinstance(config, CN):
        imgs_dir_p = Path(config.DATA.IMGS_DIR)
        metadatas_path_p = Path(config.DATA.METADATAS_PATH)
        splits_train_path_p = Path(config.DATA.SPLITS_TRAIN_PATH)
        splits_test_path_p = Path(config.DATA.SPLITS_TEST_PATH)
    else:
        imgs_dir_p = Path(imgs_dir)
        metadatas_path_p = Path(metadatas_path)
        splits_train_path_p = Path(splits_train_path)
        splits_test_path_p = Path(splits_test_path)
    removed = {
        'dish_1564159636',
        'dish_1551138237',
        'dish_1551232973',
        'dish_1551381990',
        'dish_1551389458',
        'dish_1551389551',
        'dish_1551389588',
        'dish_1551567508',
        'dish_1551567573',
        'dish_1551567604',
        'dish_1560974769'
    }
    splits_train = [line.rstrip() for line in open(splits_train_path_p,'r').readlines()]
    splits_train = list(filter(lambda t: imgs_dir_p.joinpath(t,'rgb.png').is_file() and imgs_dir_p.joinpath(t,'depth_raw.png').is_file() and t not in removed,splits_train))
    splits_test = [line.rstrip() for line in open(splits_test_path_p,'r').readlines()]
    splits_test = list(filter(lambda t: imgs_dir_p.joinpath(t,'rgb.png').is_file() and imgs_dir_p.joinpath(t,'depth_raw.png').is_file() and t not in removed,splits_test))

    if unnormalized_int_tensor or config.MODEL.NAME == 'openseed':
        transform = transforms.Compose([transforms.PILToTensor()])
        print('unnormalized')
    else:
        transform = None
    train_dataset = Nutrition5kDataset(imgs_dir_p, metadatas_path_p, splits_train, transform=transform)
    test_dataset = Nutrition5kDataset(imgs_dir_p, metadatas_path_p, splits_test, transform=transform)
    return {'train': train_dataset, 'test': test_dataset}

def collate_fn(batch):
    keys = list(batch[0].keys())
    output = {}
    new_batch = zip(*map(lambda t: t.values(), batch))
    for i, sample in enumerate(new_batch):
        if keys[i] != 'metadata':
            output[keys[i]] = default_collate(list(sample))
        else:
            output[keys[i]] = list(sample)
    return output




  


        



        
            

