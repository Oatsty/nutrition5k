import random
from dataset import Metadata, collate_fn, make_dataset
from model import SimpleInceptionV2, get_model
import init_config

from timm.scheduler import CosineLRScheduler
import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.optim.lr_scheduler import LRScheduler
from torchvision import transforms

from tqdm import tqdm
import copy
import os
from pathlib import Path
import logging
from yacs.config import CfgNode as CN

logger = logging.getLogger()

def train(config: CN, model: nn.Module, criterion: nn.Module, optimizer: optim.Optimizer, scheduler: list[LRScheduler | CosineLRScheduler], device='cuda'):
  print(' '.join(config.TITLE))
  num_epochs = config.TRAIN.NUM_EPOCHS
  batch_size = config.TRAIN.BATCH_SIZE
  resize_sizes = [(256, 352), (288, 384), (320, 448), (352, 480), (384, 512)] 

  # データセットの準備
  dataset = make_dataset(config)
  dataloader = {x: DataLoader(dataset[x], batch_size=batch_size, num_workers=8, shuffle=True if x == 'train' else False, collate_fn=collate_fn) for x in ['train','test']}

  for epoch in range(num_epochs):
      print(f'lr = {optimizer.param_groups[0]["lr"]}')
      for phase in ['train', 'test']:
          logger.info(f'Epoch {epoch+1}/{num_epochs}')
          running_loss = 0.0
          running_loss_multi = {x: 0.0 for x in ['cal','mass','fat','carb','protein']}

          # 訓練
          if phase == 'train':
            model.train()
          if phase == 'test':
             model.eval()
          print(phase)
          with torch.set_grad_enabled(phase == 'train'):
            for batch in tqdm(dataloader[phase]):
                rgb_img = batch['rgb_img']
                depth_img = batch['depth_img']
                metadata: list[Metadata] = batch['metadata']
                rgb_img = rgb_img.to(device)
                depth_img = depth_img.to(device)
                resize = transforms.Resize(random.choice(resize_sizes))
                rgb_img = resize(rgb_img)
                depth_img = resize(depth_img)

                optimizer.zero_grad()
                outputs = model(rgb_img, depth_img)
                loss_multi = {x: 0.0 for x in ['cal','mass','fat','carb','protein']}
                loss = torch.tensor(0.)
                for key in loss_multi.keys():
                    target = torch.tensor([met.__getattribute__(key) for met in metadata]).to(device)
                    loss_multi[key] = criterion(outputs[key].squeeze(), target)
                    loss = loss + loss_multi[key]
                if phase == 'train':
                    loss.backward()
                    optimizer.step()
                # print(loss.item())
                running_loss += loss.item() * len(rgb_img)
                for key in running_loss_multi.keys():
                   running_loss_multi[key] += loss_multi[key] * len(rgb_img)

          running_loss /= len(dataset[phase])
          logger.info(f'{phase} loss: {running_loss:.4f}')
          for key in running_loss_multi.keys():
             mean = dataset[phase].mean_metadata.__getattribute__(key)
             std = dataset[phase].std_metadata.__getattribute__(key)
             running_loss_multi[key] /= len(dataset[phase])
             logger.info(f'{key} loss: {running_loss_multi[key] * std:.4f}')
             logger.info(f'{key} percent loss: {running_loss_multi[key] * std / mean:.4f}')

      for s in scheduler:
          s.step(epoch+1) 

  # モデルの保存
  torch.save(model.state_dict(), config.SAVE_PATH)

def main():
  _, config = init_config.get_arguments()
  log_path = os.path.join('log',os.path.splitext(config.SAVE_PATH)[0] + '.txt')
  os.makedirs(os.path.dirname(log_path), exist_ok=True)
  logger = init_config.init_logger(os.path.dirname(log_path), os.path.basename(log_path))

  init_config.set_random_seed(config.TRAIN.SEED)
  logger.info(config.dump())
  dump_path = os.path.join('config',os.path.splitext(config.SAVE_PATH)[0] + '.yaml')
  os.makedirs(os.path.dirname(dump_path), exist_ok=True)
  with open(dump_path,'w') as f:
    f.write(config.dump())

  # モデルの準備
  device = 'cuda' if torch.cuda.is_available() else 'cpu'
  model = get_model(config,device)
  model.to(device)
  if config.TRAIN.CKPT: 
        model.load_state_dict(torch.load(config.TRAIN.CKPT))
  # 損失関数と最適化関数の定義
  criterion = nn.L1Loss()
  # オプティマイザの定義
  optimizer = optim.Adam(model.parameters(), lr=config.TRAIN.LR, weight_decay=config.TRAIN.WEIGHT_DECAY)
  # scheduler1 = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.TRAIN.NUM_EPOCHS)
  # scheduler2 = optim.lr_scheduler.MultiStepLR(optimizer,[10],gamma=10.)
  warmup_scheduler = CosineLRScheduler(optimizer, t_initial=config.TRAIN.NUM_EPOCHS - 20, warmup_t=20, warmup_lr_init=config.TRAIN.LR / 10, warmup_prefix=True)

  train(config, model, criterion, optimizer, [warmup_scheduler], device=device)


if __name__ == '__main__':
   main()