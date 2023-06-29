# --------------------------------------------------------
# X-Decoder -- Generalized Decoding for Pixel, Image, and Language
# Copyright (c) 2022 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Xueyan Zou (xueyan@cs.wisc.edu)
# --------------------------------------------------------

import logging
import os
import sys

pth = "/".join(sys.path[0].split("/")[:-1])
sys.path.insert(0, pth)

import numpy as np
from PIL import Image

np.random.seed(1)

import torch
from detectron2.data import MetadataCatalog
from detectron2.structures import BitMasks
from detectron2.utils.colormap import random_color
from openseed import build_model
from openseed.BaseModel import BaseModel
from torchvision import transforms
from utils.arguments import load_opt_command
from utils.distributed import init_distributed
from utils.visualizer import Visualizer

logger = logging.getLogger(__name__)


def main(args=None):
    """
    Main execution point for PyLearn.
    """
    opt, cmdline_args = load_opt_command(args)
    if cmdline_args.user_dir:
        absolute_user_dir = os.path.abspath(cmdline_args.user_dir)
        opt["user_dir"] = absolute_user_dir
    # Note: this threshold is lower than ordinary threshold to handle unseen objects with low confidence scores.
    threshold = 0.1
    opt = init_distributed(opt)

    # META DATA
    pretrained_pth = os.path.join(opt["WEIGHT"])
    output_root = "./output"
    image_pth = cmdline_args.image_path

    model = (
        BaseModel(opt, build_model(opt)).from_pretrained(pretrained_pth).eval().cuda()
    )

    t = []
    t.append(transforms.Resize(512, interpolation=Image.BICUBIC))
    transform = transforms.Compose(t)

    thing_classes = [
        "car",
        "person",
        "traffic light",
        "truck",
        "motorcycle",
        "cheetah",
        "jellyfish",
        "parachute",
    ]
    thing_colors = [
        random_color(rgb=True, maximum=255).astype(np.int).tolist()
        for _ in range(len(thing_classes))
    ]
    thing_dataset_id_to_contiguous_id = {x: x for x in range(len(thing_classes))}

    MetadataCatalog.get("demo").set(
        thing_colors=thing_colors,
        thing_classes=thing_classes,
        thing_dataset_id_to_contiguous_id=thing_dataset_id_to_contiguous_id,
    )

    model.model.sem_seg_head.predictor.lang_encoder.get_text_embeddings(
        thing_classes, is_eval=False
    )
    metadata = MetadataCatalog.get("demo")
    model.model.metadata = metadata
    model.model.sem_seg_head.num_classes = len(thing_classes)

    with torch.no_grad():
        image_ori = Image.open(image_pth).convert("RGB")
        width = image_ori.size[0]
        height = image_ori.size[1]
        image = transform(image_ori)
        image = np.asarray(image)
        image_ori = np.asarray(image_ori)
        images = torch.from_numpy(image.copy()).permute(2, 0, 1).cuda()

        batch_inputs = [{"image": images, "height": height, "width": width}]
        outputs = model.forward(batch_inputs, "inst_seg")
        visual = Visualizer(image_ori, metadata=metadata)
        inst_seg = outputs[-1]["instances"]
        inst_seg.pred_masks = inst_seg.pred_masks.cpu()
        inst_seg.pred_boxes = BitMasks(inst_seg.pred_masks > 0).get_bounding_boxes()
        demo = visual.draw_instance_predictions(
            inst_seg, threshold=threshold
        )  # rgb Image

        if not os.path.exists(output_root):
            os.makedirs(output_root)
        demo.save(
            os.path.join(
                output_root,
                image_pth.split("/")[-1].split(".")[0] + "_output.png",
            )
        )


if __name__ == "__main__":
    main()
    sys.exit(0)
