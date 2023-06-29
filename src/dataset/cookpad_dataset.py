import json
from pathlib import Path
from typing import Any, Optional
from yacs.config import CfgNode as CN

import pandas as pd
import numpy as np

from PIL import Image
from torchvision import transforms
import torchvision.transforms.functional as TF

from .base_dataset import BaseDataset, Metadata, Ingr


class CookpadDataset(BaseDataset):
    def __init__(self, *args, **kwargs) -> None:
        super(CookpadDataset, self).__init__(*args, **kwargs)
        self.metadatas_dict = json.load(
            open("metadata/cookpad/metadatas_refined_path.json")
        )
        self.mean_metadata = Metadata("mean", 0.0, 0.0, 0.0, 0.0, 0.0)
        self.std_metadata = Metadata("std", 1.0, 1.0, 1.0, 1.0, 1.0)

    def get_metadata(self, dish_id: str):
        path = self.metadatas_dict[dish_id]
        mean_metadata = self.mean_metadata
        std_metadata = self.std_metadata
        df = pd.read_csv(path)
        data_dict = Metadata()
        data_dict.dish_id = dish_id
        data_dict.cal = (
            df["ENERC_KCAL"].dropna().sum() - mean_metadata.cal
        ) / std_metadata.cal
        data_dict.mass = (
            df["Weight"].dropna().sum() - mean_metadata.mass
        ) / std_metadata.mass
        data_dict.fat = (
            df["FAT-"].dropna().sum() - mean_metadata.fat
        ) / std_metadata.fat
        data_dict.carb = (
            df["CHOAVLDF-"].dropna().sum() - mean_metadata.carb
        ) / std_metadata.carb
        data_dict.protein = (
            df["PROT-"].dropna().sum() - mean_metadata.protein
        ) / std_metadata.protein
        data_dict.ingrs = []
        for _, row in df.iterrows():  # type: ignore
            ingr = Ingr()
            ingr.id = row["food_item"]
            ingr.name = row["Ingr_Exp"]
            ingr.cal = (row["ENERC_KCAL"] - mean_metadata.cal) / std_metadata.cal
            ingr.grams = (row["Weight"] - mean_metadata.mass) / std_metadata.mass
            ingr.fat = (row["FAT-"] - mean_metadata.fat) / std_metadata.fat
            ingr.carb = (row["CHOAVLDF-"] - mean_metadata.carb) / std_metadata.carb
            ingr.protein = (row["PROT-"] - mean_metadata.protein) / std_metadata.protein
            data_dict.ingrs.append(ingr)
        return data_dict

    def __getitem__(self, index: int) -> dict[str, Any]:
        img_path = Path.joinpath(self.imgs_dir, self.splits[index])
        rgb_img = Image.open(img_path)
        rgb_img = self.transform(rgb_img)
        rgb_img, depth_img = self.transform_(rgb_img, rgb_img)
        metadata = self.get_metadata(Path(self.splits[index]).stem)
        sample = {
            "rgb_img": rgb_img,
            "depth_img": depth_img,
            "metadata": metadata,
            "rgb_path": str(img_path),
            "depth_path": str(img_path),
        }
        return sample

    @staticmethod
    def transform_(rgb_img, depth_img):
        # rotate
        params = transforms.RandomRotation.get_params([-180, 180])
        rgb_img = TF.rotate(rgb_img, params)
        depth_img = TF.rotate(depth_img, params)

        if np.random.rand() < 0.5:
            rgb_img = TF.hflip(rgb_img)
            depth_img = TF.hflip(depth_img)
        return rgb_img, depth_img


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
    splits_train = [
        line.rstrip() for line in open(splits_train_path_p, "r").readlines()
    ]
    splits_test = [line.rstrip() for line in open(splits_test_path_p, "r").readlines()]

    model_name = config.MODEL.NAME if config != None else ""

    if unnormalized_int_tensor or model_name == "openseed":
        print("unnormalized")
        transform = transforms.Compose([transforms.PILToTensor()])
        normalize_depth = transforms.Normalize(0, 5.361)
    else:
        transform = None
        normalize_depth = None
    train_dataset = CookpadDataset(
        imgs_dir_p,
        metadatas_path_p,
        splits_train,
        transform=transform,
        normalize_depth=normalize_depth,
    )
    test_dataset = CookpadDataset(
        imgs_dir_p,
        metadatas_path_p,
        splits_test,
        transform=transform,
        normalize_depth=normalize_depth,
    )
    return {"train": train_dataset, "test": test_dataset}
