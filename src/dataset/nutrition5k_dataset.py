from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torchvision import transforms
from yacs.config import CfgNode as CN

from .base_dataset import BaseDataset, Ingr, Metadata


class Nutrition5kDataset(BaseDataset):
    """
    Nutrition 5k Dataset Class

    Args:
        imgs_dir (Path): Path to images directory
        metadatas_path (Path): Path to metadatas (csv file)
        splits (list[str]): list of image names in current split
        transform (Optional[transforms.Compose]): RGB image transforms (default=None)
        transform_depth (Optional[transforms.Compose]): depth image transforms (default=None)
        normalize_depth (Optional[transforms.Compose]): dpeth normalization (default=None)
        rotate_flip (Optional[bool]): Whether to perform rotation and horizontal flip data augmentation (default=True)
        w_depth (Optional[bool]): Whether to load depth data (default=True)
        w_mask (Optional[bool]): Whether to load top-1 mask (default=True)
        w_mask_all (Optional[bool]): Whether to load all masks (default=True)
    """
    def __init__(self, *args, **kwargs) -> None:
        super(Nutrition5kDataset, self).__init__(*args, **kwargs)
        self.metadatas_dict: dict[str, Metadata] = {}
        self.init_metadatas()

    @staticmethod
    def transform_(imgs: dict[str, Any]) -> dict[str, torch.Tensor]:
        # rotate
        params = transforms.RandomRotation.get_params([-180, 180])
        imgs = {k: TF.rotate(img, params) for k, img in imgs.items()}

        if np.random.rand() < 0.5:
            imgs = {k: TF.hflip(img) for k, img in imgs.items()}
        return imgs

    def init_metadatas(self):
        """
        Extract metadatas for all dishes and normalize
        """
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

    def __getitem__(self, index: int) -> dict[str, Any]:
        metadata = self.metadatas_dict[self.splits[index]]
        img_dir = Path.joinpath(self.imgs_dir, self.splits[index])
        rgb_path = Path.joinpath(img_dir, "rgb.png")
        depth_path = Path.joinpath(img_dir, "depth_raw.png")
        mask_path = Path.joinpath(img_dir, "mask.pt")
        mask_all_path = Path.joinpath(img_dir, "mask_all.pt")
        sample = {
            "metadata": metadata,
            "rgb_path": str(rgb_path),
            "depth_path": str(depth_path),
        }
        imgs = {}
        rgb_img = Image.open(rgb_path)
        rgb_img = self.transform(rgb_img)
        imgs["rgb_img"] = rgb_img
        if self.w_depth:
            depth_img = Image.open(depth_path)
            depth_img = self.transform_depth(depth_img).float()  # type: ignore
            depth_img = self.normalize_depth(depth_img)
            imgs["depth_img"] = depth_img
        if self.w_mask:
            mask = torch.load(mask_path)
            mask = mask.unsqueeze(0)
            imgs["mask"] = mask
        if self.w_mask_all:
            imgs["mask_all"] = torch.load(mask_all_path)

        if self.rotate_flip:
            imgs = self.transform_(imgs)
        if self.w_mask:
            imgs["mask"] = imgs["mask"].squeeze(0)
        sample.update(imgs)
        return sample


def make_dataset(
    config: Optional[CN],
    imgs_dir: str = ".",
    metadatas_path: str = ".",
    splits_train_path: str = ".",
    splits_test_path: str = ".",
    unnormalized_int_tensor: bool = False,
    **kwargs,
) -> dict[str, BaseDataset]:
    if isinstance(config, CN):
        imgs_dir_p = Path(config.DATA.IMGS_DIR)
        metadatas_path_p = Path(config.DATA.METADATAS_PATH)
        splits_train_path_p = Path(config.DATA.SPLITS_TRAIN_PATH)
        splits_test_path_p = Path(config.DATA.SPLITS_TEST_PATH)
        if config.TRAIN.TRAINER != "multi_mask":
            kwargs.update(w_mask_all=False)
        if config.TRAIN.TRAINER == "normal" or config.TRAIN.TRAINER == "noisy":
            kwargs.update(w_mask=False)
    else:
        imgs_dir_p = Path(imgs_dir)
        metadatas_path_p = Path(metadatas_path)
        splits_train_path_p = Path(splits_train_path)
        splits_test_path_p = Path(splits_test_path)

    ### dishes with obvious mislabels
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

    model_name = config.MODEL.NAME if config is not None else ""

    if unnormalized_int_tensor or model_name == "openseed":
        print("unnormalized")
        transform = transforms.Compose([transforms.PILToTensor()])
        normalize_depth = transforms.Normalize(3091, 5.361)
    else:
        transform = None
        normalize_depth = None
    train_dataset = Nutrition5kDataset(
        imgs_dir_p,
        metadatas_path_p,
        splits_train,
        transform=transform,
        normalize_depth=normalize_depth,
        **kwargs,
    )
    test_dataset = Nutrition5kDataset(
        imgs_dir_p,
        metadatas_path_p,
        splits_test,
        transform=transform,
        normalize_depth=normalize_depth,
        **kwargs,
    )
    return {"train": train_dataset, "test": test_dataset}
