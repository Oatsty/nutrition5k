# --------------------------------------------------------
# X-Decoder -- Generalized Decoding for Pixel, Image, and Language
# Copyright (c) 2022 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Xueyan Zou (xueyan@cs.wisc.edu)
# --------------------------------------------------------

import os
import sys
import logging

pth = '/'.join(sys.path[0].split('/')[:-1])
sys.path.insert(0, pth)

from PIL import Image
import numpy as np
np.random.seed(1)

import torch
from torchvision import transforms

from utils.arguments import load_opt_command

from detectron2.data import MetadataCatalog
from detectron2.utils.colormap import random_color
from openseed.BaseModel import BaseModel
from openseed import build_model
from utils.visualizer import Visualizer
from utils.distributed import init_distributed

logger = logging.getLogger(__name__)


def main(args=None):
    '''
    Main execution point for PyLearn.
    '''
    opt, cmdline_args = load_opt_command(args)
    if cmdline_args.user_dir:
        absolute_user_dir = os.path.abspath(cmdline_args.user_dir)
        opt['user_dir'] = absolute_user_dir
    # Note: this threshold is lower than ordinary threshold to handle unseen objects with low confidence scores.
    opt['MODEL']['DECODER']['TEST']['OBJECT_MASK_THRESHOLD'] = 0.08
    opt = init_distributed(opt)

    # META DATA
    pretrained_pth = os.path.join(opt['WEIGHT'])
    output_root = './output'
    image_pth = cmdline_args.image_path

    model = BaseModel(opt, build_model(opt)).from_pretrained(pretrained_pth).eval().cuda()

    t = []
    t.append(transforms.Resize(512, interpolation=Image.BICUBIC))
    transform = transforms.Compose(t)

    # add your own vocabulary if necessary
    thing_classes = ['car','person', 'truck', 'motorcycle', 'cheetah', 'jellyfish', 'butterfly', 'parachute']
    stuff_classes = ['street','rock']
    thing_colors = [random_color(rgb=True, maximum=255).astype(np.int).tolist() for _ in range(len(thing_classes))]
    stuff_colors = [random_color(rgb=True, maximum=255).astype(np.int).tolist() for _ in range(len(stuff_classes))]
    thing_dataset_id_to_contiguous_id = {x:x for x in range(len(thing_classes))}
    stuff_dataset_id_to_contiguous_id = {x+len(thing_classes):x for x in range(len(stuff_classes))}

    MetadataCatalog.get("demo").set(
        thing_colors=thing_colors,
        thing_classes=thing_classes,
        thing_dataset_id_to_contiguous_id=thing_dataset_id_to_contiguous_id,
        stuff_colors=stuff_colors,
        stuff_classes=stuff_classes,
        stuff_dataset_id_to_contiguous_id=stuff_dataset_id_to_contiguous_id,
    )
    model.model.sem_seg_head.predictor.lang_encoder.get_text_embeddings(thing_classes + stuff_classes, is_eval=False)
    metadata = MetadataCatalog.get('demo')
    model.model.metadata = metadata
    model.model.sem_seg_head.num_classes = len(thing_classes + stuff_classes)

    with torch.no_grad():
        image_ori = Image.open(image_pth).convert("RGB")
        width = image_ori.size[0]
        height = image_ori.size[1]
        image = transform(image_ori)
        image = np.asarray(image)
        image_ori = np.asarray(image_ori)
        images = torch.from_numpy(image.copy()).permute(2,0,1).cuda()

        batch_inputs = [{'image': images, 'height': height, 'width': width}]
        outputs = model.forward(batch_inputs, 'pan_seg')
        visual = Visualizer(image_ori, metadata=metadata)

        pano_seg = outputs[-1]['panoptic_seg'][0]
        pano_seg_info = outputs[-1]['panoptic_seg'][1]

        for i in range(len(pano_seg_info)):
            if pano_seg_info[i]['category_id'] in metadata.thing_dataset_id_to_contiguous_id.keys():
                pano_seg_info[i]['category_id'] = metadata.thing_dataset_id_to_contiguous_id[pano_seg_info[i]['category_id']]
            else:
                pano_seg_info[i]['isthing'] = False
                pano_seg_info[i]['category_id'] = metadata.stuff_dataset_id_to_contiguous_id[pano_seg_info[i]['category_id']]

        demo = visual.draw_panoptic_seg(pano_seg.cpu(), pano_seg_info) # rgb Image

        if not os.path.exists(output_root):
            os.makedirs(output_root)
        demo.save(os.path.join(output_root, image_pth.split('/')[-1].split('.')[0] + '_output.png'))


if __name__ == "__main__":
    main()
    sys.exit(0)