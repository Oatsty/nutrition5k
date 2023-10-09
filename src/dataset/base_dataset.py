from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from torch.utils.data import Dataset, default_collate
from torchvision import transforms


@dataclass
class Ingr:
    id: str = field(default_factory=str)
    name: str = field(default_factory=str)
    grams: float = field(default_factory=float)
    cal: float = field(default_factory=float)
    fat: float = field(default_factory=float)
    carb: float = field(default_factory=float)
    protein: float = field(default_factory=float)

    def __str__(self) -> str:
        ingr_dict = {
            "id": self.id,
            "name": self.name,
            "grams": self.grams,
            "cal": self.cal,
            "fat": self.fat,
            "carb": self.carb,
            "protein": self.protein,
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
            "dish_id": self.dish_id,
            "cal": self.cal,
            "mass": self.mass,
            "fat": self.fat,
            "carb": self.carb,
            "protein": self.protein,
            "ingrs": self.ingrs,
        }
        return str(metadata_dict)


def collate_fn(batch):
    keys = list(batch[0].keys())
    output = {}
    new_batch = zip(*map(lambda t: t.values(), batch))
    for i, sample in enumerate(new_batch):
        if keys[i] != "metadata":
            output[keys[i]] = default_collate(list(sample))
        else:
            output[keys[i]] = list(sample)
    return output


class BaseDataset(Dataset):
    """
    Base Dataset Class

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
    def __init__(
        self,
        imgs_dir: Path,
        metadatas_path: Path,
        splits: list[str],
        transform: Optional[transforms.Compose] = None,
        transform_depth: Optional[transforms.Compose] = None,
        normalize_depth: Optional[transforms.Compose] = None,
        rotate_flip: bool = True,
        w_depth: bool = True,
        w_mask: bool = True,
        w_mask_all: bool = True,
    ) -> None:
        super(BaseDataset, self).__init__()
        self.imgs_dir = imgs_dir
        self.metadatas_path = metadatas_path
        self.splits = splits
        self.rotate_flip = rotate_flip
        self.w_depth = w_depth
        self.w_mask = w_mask
        self.w_mask_all = w_mask_all
        self.mean_metadata = Metadata("mean", 255.0, 218.0, 12.7, 19.3, 18.1)
        self.std_metadata = Metadata("std", 221.0, 163.0, 13.4, 22.3, 20.2)
        self.metadatas_dict = {}
        if isinstance(transform, transforms.Compose):
            self.transform = transform
        else:
            self.transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ]
            )
        if isinstance(transform_depth, transforms.Compose):
            self.transform_depth = transform_depth
        else:
            self.transform_depth = transforms.Compose(
                [
                    transforms.PILToTensor(),
                ]
            )
        if isinstance(normalize_depth, transforms.Normalize):
            self.normalize_depth = normalize_depth
        else:
            self.normalize_depth = transforms.Normalize(3091, 1307)

    def __len__(self):
        return len(self.splits)
