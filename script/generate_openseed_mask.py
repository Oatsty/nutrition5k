import sys
from pathlib import Path

pth = "/".join(sys.path[0].split("/")[:-1])
openseed_pth = "/".join([pth, "OpenSeeD"])
sys.path.insert(0, openseed_pth)
sys.path.append(pth)

import torch
import argparse
from src.dataset import collate_fn
from src.dataset.nutrition5k_dataset import make_dataset
from src.seg_openseed import OpenSeeDSeg
from torch.utils.data import DataLoader
from tqdm import tqdm


def main():
    # arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('dir', type=str, help='dataset directory')
    args = parser.parse_args()
    data_dir = args.dir
    metadatas_path = (
        f"{data_dir}/metadata/dish_metadata_cafe1.csv"
    )
    imgs_dir = f"{data_dir}/imagery/realsense_overhead"
    splits_path = (
        f"{data_dir}/dish_ids/splits/depth_train_ids.txt"
    )
    splits_test_path = (
        f"{data_dir}/dish_ids/splits/depth_test_ids.txt"
    )
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # Init OpenSeeD segmentation module
    openseed_seg = OpenSeeDSeg(device)

    # Init dataset
    dataset = make_dataset(
        None,
        imgs_dir,
        metadatas_path,
        splits_path,
        splits_test_path,
        unnormalized_int_tensor=True,
    )
    dataloader = {
        x: DataLoader(
            dataset[x],
            batch_size=32,
            num_workers=8,
            shuffle=False,
            collate_fn=collate_fn,
        )
        for x in ["train", "test"]
    }

    # Generate food region mask for all images
    for phase in ["train", "test"]:
        for batch in tqdm(dataloader[phase]):
            rgb_img = batch["rgb_img"]
            depth_img = batch["depth_img"]
            rgb_path = batch["rgb_path"]
            rgb_img = rgb_img.to(device)
            depth_img = depth_img.to(device)
            masks, _, _ = openseed_seg.get_mask(rgb_img)
            masks = masks.bool().cpu()
            for rgbp, m in zip(rgb_path, masks):
                maskp = Path(rgbp).parent.joinpath("mask.pt")
                torch.save(m, maskp)


if __name__ == "__main__":
    main()
