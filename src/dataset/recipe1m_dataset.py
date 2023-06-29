from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from yacs.config import CfgNode as CN

from .base_dataset import BaseDataset, Ingr, Metadata


class Recipe1MDataset(BaseDataset):
    def __init__(self, *args, **kwargs) -> None:
        super(Recipe1MDataset, self).__init__(*args, **kwargs)
        self.metadatas_dict: dict[str, Metadata] = {}
        self.init_metadatas()

    @staticmethod
    def transform_(rgb_img, depth_img):
        # rotate
        params = transforms.RandomRotation.get_params([-180, 180])
        rgb_img = TF.rotate(rgb_img, params)
        depth_img = TF.rotate(depth_img, params, fill=depth_img.max().item())

        if np.random.rand() < 0.5:
            rgb_img = TF.hflip(rgb_img)
            depth_img = TF.hflip(depth_img)
        return rgb_img, depth_img

    def init_metadatas(self):
        mean_metadata = self.mean_metadata
        std_metadata = self.std_metadata
        for line in open(self.metadatas_path, "r").readlines():
            line = line.rstrip()
            data_list = line.split(",")
            data_dict = Metadata()
            dish_id = data_list[0]
            data_dict.dish_id = dish_id
            data_dict.cal = (float(data_list[1]) - mean_metadata.cal) / std_metadata.cal
            data_dict.mass = (
                float(data_list[2]) - mean_metadata.mass
            ) / std_metadata.mass
            data_dict.fat = (float(data_list[3]) - mean_metadata.fat) / std_metadata.fat
            data_dict.carb = (
                float(data_list[4]) - mean_metadata.carb
            ) / std_metadata.carb
            data_dict.protein = (
                float(data_list[5]) - mean_metadata.protein
            ) / std_metadata.protein
            data_dict.ingrs = []
            for id, name, grams, cal, fat, carb, protein in zip(*[data_list[x::7] for x in range(6, 13)]):  # type: ignore
                ingr = Ingr(
                    **{
                        "id": id,
                        "name": name,
                        "grams": grams,
                        "cal": cal,
                        "fat": fat,
                        "carb": carb,
                        "protein": protein,
                    }
                )
                data_dict.ingrs.append(ingr)
            assert len(data_list) == len(data_dict.ingrs) * 7 + 6
            self.metadatas_dict[dish_id] = data_dict

    def __len__(self):
        return len(self.splits)

    def __getitem__(self, index: int) -> dict[str, Any]:
        img_dir = Path.joinpath(self.imgs_dir, self.splits[index])
        rgb_path = Path.joinpath(img_dir, "rgb.png")
        depth_path = Path.joinpath(img_dir, "depth_raw.png")
        rgb_img = Image.open(rgb_path)
        depth_img = Image.open(depth_path)
        rgb_img = self.transform(rgb_img)
        depth_img = self.transform_depth(depth_img).float()  # type: ignore
        depth_img = self.normalize_depth(depth_img)
        rgb_img, depth_img = self.transform_(rgb_img, depth_img)
        metadata = self.metadatas_dict[self.splits[index]]
        sample = {
            "rgb_img": rgb_img,
            "depth_img": depth_img,
            "metadata": metadata,
            "rgb_path": str(rgb_path),
            "depth_path": str(depth_path),
        }
        return sample


def make_dataset(
    config: Optional[CN],
    imgs_dir: str = ".",
    metadatas_path: str = ".",
    splits_train_path: str = ".",
    splits_test_path: str = ".",
    unnormalized_int_tensor: bool = False,
) -> dict[str, BaseDataset]:
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
        "dish_1564159636",
        "dish_1551138237",
        "dish_1551232973",
        "dish_1551381990",
        "dish_1551389458",
        "dish_1551389551",
        "dish_1551389588",
        "dish_1551567508",
        "dish_1551567573",
        "dish_1551567604",
        "dish_1560974769",
    }
    splits_train = [
        line.rstrip() for line in open(splits_train_path_p, "r").readlines()
    ]
    splits_train = list(
        filter(
            lambda t: imgs_dir_p.joinpath(t, "rgb.png").is_file()
            and imgs_dir_p.joinpath(t, "depth_raw.png").is_file()
            and t not in removed,
            splits_train,
        )
    )
    splits_test = [line.rstrip() for line in open(splits_test_path_p, "r").readlines()]
    splits_test = list(
        filter(
            lambda t: imgs_dir_p.joinpath(t, "rgb.png").is_file()
            and imgs_dir_p.joinpath(t, "depth_raw.png").is_file()
            and t not in removed,
            splits_test,
        )
    )

    model_name = config.MODEL.NAME if config != None else ""

    if unnormalized_int_tensor or model_name == "openseed":
        print("unnormalized")
        transform = transforms.Compose([transforms.PILToTensor()])
        normalize_depth = transforms.Normalize(0, 5.361)
    else:
        transform = None
        normalize_depth = None
    train_dataset = Recipe1MDataset(
        imgs_dir_p,
        metadatas_path_p,
        splits_train,
        transform=transform,
        normalize_depth=normalize_depth,
    )
    test_dataset = Recipe1MDataset(
        imgs_dir_p,
        metadatas_path_p,
        splits_test,
        transform=transform,
        normalize_depth=normalize_depth,
    )
    return {"train": train_dataset, "test": test_dataset}
