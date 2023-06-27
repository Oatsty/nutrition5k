from pathlib import Path
import sys

sys.path.insert(0,'/home/parinayok/nutrition5k/OpenSeeD')
sys.path.append('/home/parinayok/nutrition5k')

from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

from src.seg_openseed import OpenSeeDSeg
from src.dataset import collate_fn
from src.dataset.nutrition5k_dataset import make_dataset

def main():
  metadatas_path = '/srv/datasets2/nutrition5k_dataset/metadata/dish_metadata_cafe1.csv'
  imgs_dir = '/srv/datasets2/nutrition5k_dataset/imagery/realsense_overhead'
  splits_path = '/srv/datasets2/nutrition5k_dataset/dish_ids/splits/depth_train_ids.txt'
  splits_test_path = '/srv/datasets2/nutrition5k_dataset/dish_ids/splits/depth_test_ids.txt'
  device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
  openseed_seg = OpenSeeDSeg(device)
  dataset = make_dataset(None,imgs_dir,metadatas_path,splits_path,splits_test_path,unnormalized_int_tensor=True)
  dataloader = {x: DataLoader(dataset[x], batch_size=32, num_workers=8, shuffle=False, collate_fn=collate_fn) for x in ['train','test']}
  for phase in ['train','test']:
    for batch in tqdm(dataloader[phase]):
      rgb_img = batch['rgb_img']
      depth_img = batch['depth_img']
      rgb_path = batch['rgb_path']
      rgb_img = rgb_img.to(device)
      depth_img = depth_img.to(device)
      masks, _, _ = openseed_seg.get_mask(rgb_img)
      masks = masks.bool().cpu()
      for rgbp, m in zip(rgb_path, masks):
        maskp = Path(rgbp).parent.joinpath('mask.pt')
        torch.save(m,maskp)

if __name__ == '__main__':
   main()