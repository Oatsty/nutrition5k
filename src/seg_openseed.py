import os
import sys
import logging

sys.path.append('..')
sys.path.append('OpenSeeD')

from PIL import Image
import numpy as np

import torch
from torchvision import transforms

from OpenSeeD.utils.arguments import load_opt_from_config_files

from detectron2.data import MetadataCatalog
from detectron2.utils.colormap import random_color
from OpenSeeD.openseed.BaseModel import BaseModel
from OpenSeeD.openseed import build_model
from OpenSeeD.utils.visualizer import Visualizer
from OpenSeeD.utils.distributed import init_distributed

logger = logging.getLogger(__name__)

def seg(image: torch.Tensor,device: torch.device):
    conf_files = ['/home/parinayok/nutrition5k/OpenSeeD/configs/openseed/openseed_swint_lang.yaml']
    opt = load_opt_from_config_files(conf_files)
    opt['cont_files'] = conf_files
    opt['command'] = 'evaluate'
    opt = init_distributed(opt)

    # META DATA
    pretrained_pth = os.path.join(opt['WEIGHT'])
    model = BaseModel(opt, build_model(opt)).from_pretrained(pretrained_pth).eval().to(device)

    # t = []
    # t.append(transforms.Resize(512, interpolation=Image.BICUBIC))
    # transform = transforms.Compose(t)

    thing_classes = ['food']
    # stuff_classes = ['apple','zebra','antelope','giraffe','ostrich','sky','water','grass','sand','tree']
    model.model.sem_seg_head.predictor.lang_encoder.get_text_embeddings(thing_classes, is_eval=False) # type: ignore
    metadata = MetadataCatalog.get('demo')
    model.model.metadata = metadata # type: ignore
    model.model.sem_seg_head.num_classes = len(thing_classes) # type: ignore

    with torch.no_grad():
        height = image.shape[-2]
        width = image.shape[-1]
        # image = transform(image)
        batch_inputs = [{'image': image, 'height': height, 'width': width}]
        outputs = model.forward(batch_inputs,'inst_seg')
    return outputs
        # visual = Visualizer(image_ori, metadata=metadata)

        # sem_seg = outputs[-1]['sem_seg'].max(0)[1]
        # demo = visual.draw_sem_seg(sem_seg.cpu(), alpha=0.5) # rgb Image

        # if not os.path.exists(output_root):
        #     os.makedirs(output_root)
        # demo.save(os.path.join(output_root, 'sem.png'))


if __name__ == "__main__":
    x = torch.rand(3,256,256)
    seg(x,torch.device('cuda'))
    sys.exit(0)