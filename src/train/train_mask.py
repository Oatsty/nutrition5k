import os

import logging
import random
from tqdm import tqdm
from yacs.config import CfgNode as CN

from timm.scheduler import CosineLRScheduler
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from torchvision import transforms

from .base import BaseTrainer
from custom_utils import get_loss
from dataset import Metadata, collate_fn, make_dataset
import init_config
from model import BaseModel, get_model

logger = logging.getLogger()

class MaskTrainer(BaseTrainer):
    @staticmethod
    def train(config: CN, model: BaseModel, device: torch.device):
      print(' '.join(config.TITLE))
      num_epochs = config.TRAIN.NUM_EPOCHS
      batch_size = config.TRAIN.BATCH_SIZE
      resize_sizes = [(256, 352), (288, 384), (320, 448), (352, 480), (384, 512), (480, 640)] 
      loss_func = get_loss(config)
      optimizer = optim.Adam(model.parameters(), lr=config.TRAIN.LR, weight_decay=config.TRAIN.WEIGHT_DECAY)
      scheduler = CosineLRScheduler(optimizer, t_initial=config.TRAIN.NUM_EPOCHS - 20, lr_min=1e-7, warmup_t=20, warmup_prefix=True)

      # データセットの準備
      dataset = make_dataset(config)
      dataloader = {x: DataLoader(dataset[x], batch_size=batch_size, num_workers=8, shuffle=True if x == 'train' else False, collate_fn=collate_fn) for x in ['train','test']}

      for epoch in range(num_epochs):
          print(f'lr = {optimizer.param_groups[0]["lr"]}')
          for phase in ['train', 'test']:
              logger.info(f'Epoch {epoch+1}/{num_epochs}')
              running_loss = 0.0
              running_loss_multi = {}

              # 訓練
              if phase == 'train':
                model.train()
              if phase == 'test':
                model.eval()
              print(phase)
              with torch.set_grad_enabled(phase == 'train'):
                for batch in tqdm(dataloader[phase]):
                    rgb_img: torch.Tensor = batch['rgb_img']
                    depth_img: torch.Tensor = batch['depth_img']
                    mask: torch.Tensor = batch['mask']
                    metadata: list[Metadata] = batch['metadata']
                    rgb_img = rgb_img.to(device)
                    depth_img = depth_img.to(device)
                    mask = mask.to(device)
                    if phase == 'train':
                      resize = transforms.Resize(random.choice(resize_sizes))
                    else:
                      resize = transforms.Resize(resize_sizes[-1])
                    rgb_img = resize(rgb_img)
                    depth_img = resize(depth_img)
                    mask = resize(mask)

                    optimizer.zero_grad()
                    outputs = model(rgb_img, depth_img, mask=mask)
                    loss_multi = loss_func(outputs,metadata,device)
                    loss = sum(loss_multi.values())
                    assert isinstance(loss,torch.Tensor)
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                    # print(loss.item())
                    running_loss += loss.item() * len(rgb_img)
                    for key in loss_multi.keys():
                        if key in running_loss_multi.keys():
                            running_loss_multi[key] += loss_multi[key].item() * len(rgb_img)
                        else:
                            running_loss_multi[key] = loss_multi[key].item() * len(rgb_img)

              running_loss /= len(dataset[phase])
              logger.info(f'{phase} loss: {running_loss:.4f}')
              for key in running_loss_multi.keys():
                  running_loss_multi[key] /= len(dataset[phase])
                  if key == 'ingrs':
                    logger.info(f'{key} percent loss: {running_loss_multi[key]:.4f}')
                    continue
                  mean = dataset[phase].mean_metadata.__getattribute__(key)
                  std = dataset[phase].std_metadata.__getattribute__(key)
                  logger.info(f'{key} loss: {running_loss_multi[key] * std:.4f}')
                  logger.info(f'{key} percent loss: {running_loss_multi[key] * std / mean:.4f}')

          scheduler.step(epoch+1)

      # モデルの保存
      torch.save(model.state_dict(), config.SAVE_PATH)

def main():
  _, config = init_config.get_arguments()
  os.makedirs(os.path.dirname(config.SAVE_PATH), exist_ok=True)

  # モデルの準備
  device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
  model = get_model(config,device)
  model.to(device)
  if config.TRAIN.CKPT: 
        model.load_state_dict(torch.load(config.TRAIN.CKPT))
  
  log_path = os.path.join('log',os.path.splitext(config.SAVE_PATH)[0] + '.txt')
  os.makedirs(os.path.dirname(log_path), exist_ok=True)
  logger = init_config.init_logger(os.path.dirname(log_path), os.path.basename(log_path))

  init_config.set_random_seed(config.TRAIN.SEED)
  logger.info(config.dump())
  dump_path = os.path.join('config',os.path.splitext(config.SAVE_PATH)[0] + '.yaml')
  os.makedirs(os.path.dirname(dump_path), exist_ok=True)
  with open(dump_path,'w') as f:
    f.write(config.dump()) #type: ignore

  MaskTrainer.train(config, model, device=device)

if __name__ == '__main__':
   main()