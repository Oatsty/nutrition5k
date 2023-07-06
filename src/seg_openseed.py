import os
import sys

sys.path.append("..")
sys.path.append("OpenSeeD")

import torch
from custom_utils import MetadataCatalog, random_color
from OpenSeeD.openseed import build_model
from OpenSeeD.openseed.BaseModel import BaseModel
from OpenSeeD.utils.arguments import load_opt_from_config_files
from OpenSeeD.utils.distributed import init_distributed


class OpenSeeDSeg:
    def __init__(self, device) -> None:
        super(OpenSeeDSeg, self).__init__()
        conf_files = ["OpenSeeD/configs/openseed/openseed_swint_lang.yaml"]
        opt = load_opt_from_config_files(conf_files)
        opt["cont_files"] = conf_files
        opt["command"] = "evaluate"
        opt = init_distributed(opt)
        pretrained_pth = os.path.join(opt["WEIGHT"])
        thing_classes = ["food"]
        thing_colors = [
            random_color(rgb=True, maximum=255).astype(int).tolist()
            for _ in range(len(thing_classes))
        ]
        thing_dataset_id_to_contiguous_id = {x: x for x in range(len(thing_classes))}
        MetadataCatalog.get("demo").set(
            thing_colors=thing_colors,
            thing_classes=thing_classes,
            thing_dataset_id_to_contiguous_id=thing_dataset_id_to_contiguous_id,
        )
        metadata = MetadataCatalog.get("demo")
        print(metadata)
        self.model = (
            BaseModel(opt, build_model(opt))
            .from_pretrained(pretrained_pth)
            .eval()
            .to(device)
        )
        self.model.model.metadata = metadata  # type: ignore
        self.model.model.sem_seg_head.predictor.lang_encoder.get_text_embeddings(thing_classes, is_eval=False)  # type: ignore
        self.model.model.sem_seg_head.num_classes = len(thing_classes)  # type: ignore

    def seg(self, image: torch.Tensor):
        with torch.no_grad():
            height = image.shape[-2]
            width = image.shape[-1]
            batch_inputs = [
                {"image": img, "height": height, "width": width} for img in image
            ]
            outputs = self.model.forward(batch_inputs, "inst_seg")
        return outputs

    def get_mask(self, img: torch.Tensor, all_mask: bool = False, top: int = 20):
        outputs = self.seg(img)
        features = outputs["backbone_features"]
        mask_batch = []
        inst_seg_batch = []
        for res in outputs["results"]:
            inst_seg = res["instances"]
            inst_seg_batch.append(inst_seg)
            masks = inst_seg.pred_masks
            scores = inst_seg.scores
            if all_mask:
                masks = masks[scores.argsort(descending=True)]
                mask_batch.append(masks[:top])
                continue
            keep = scores > 0.1
            masks = masks[keep]
            if len(masks) == 0:
                mask = torch.ones(masks.shape[1], masks.shape[2], device=masks.device)
            else:
                mask = masks.max(0)[0]
            mask_batch.append(mask)
        mask_batch_tensor = torch.stack(mask_batch)
        return mask_batch_tensor, inst_seg_batch, features
