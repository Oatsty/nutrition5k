import os
import sys
sys.path.insert(0,'/home/parinayok/nutrition5k/OpenSeeD')
sys.path.append('/home/parinayok/nutrition5k')

import logging
import random
from tqdm import tqdm
from typing import Callable
from yacs.config import CfgNode as CN

from timm.scheduler import CosineLRScheduler
import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.optim.lr_scheduler import LRScheduler
from torchvision import transforms

from dataset import Metadata, collate_fn, make_dataset
from model import get_model
import init_config
from custom_utils import get_candidate, get_loss

logger = logging.getLogger()

def train(config: CN, model: nn.Module, optimizer: optim.Optimizer, scheduler: LRScheduler, device: torch.device):
  print(' '.join(config.TITLE))
  num_epochs = config.TRAIN.NUM_EPOCHS
  warmup_epochs = config.TRAIN.WARMUP_EPOCHS
  noise_removal_epochs = config.TRAIN.NOISE_REMOVAL_EPOCHS
  remove_rate = config.TRAIN.REMOVE_RATE
  batch_size = config.TRAIN.BATCH_SIZE
  resize_sizes = [(256, 352), (288, 384), (320, 448), (352, 480), (384, 512), (480, 640)]
  loss_func = get_loss(config)

  # データセットの準備
  dataset = make_dataset(config)
  dataloader = {x: DataLoader(dataset[x], batch_size=batch_size, num_workers=8, shuffle=True if x == 'train' else False, collate_fn=collate_fn) for x in ['train','test']}

  for epoch in range(num_epochs):
      if epoch < warmup_epochs:
        noisy_phase = 'warmup'
      elif epoch < warmup_epochs + noise_removal_epochs:
        noisy_phase = 'noise_removal'
      else:
        noisy_phase = 'normal'
      print(f'lr = {optimizer.param_groups[0]["lr"]}')
      for phase in ['train', 'test']:
          logger.info(f'Epoch {epoch+1}/{num_epochs}')
          print(phase)
          running_loss = 0.0
          running_loss_multi = {}

          if phase == 'train':
            model.train()
          if phase == 'test':
            model.eval()

          outputs_cands = {}
          with torch.set_grad_enabled(phase == 'train'):
            for batch in tqdm(dataloader[phase]):
                rgb_img = batch['rgb_img']
                depth_img = batch['depth_img']
                metadata: list[Metadata] = batch['metadata']
                rgb_img = rgb_img.to(device)
                depth_img = depth_img.to(device)
                if phase == 'train':
                  resize = transforms.Resize(random.choice(resize_sizes))
                else:
                  resize = transforms.Resize(resize_sizes[-1])
                rgb_img = resize(rgb_img)
                depth_img = resize(depth_img)

                optimizer.zero_grad()
                if noisy_phase == 'noise_removal' or noisy_phase == 'warmup':
                  outputs = model(rgb_img, depth_img, skip_attn=True, depth=2)
                else:
                  outputs = model(rgb_img, depth_img)
                if phase == 'train' and noisy_phase == 'noise_removal':
                  loss_multi = loss_func(outputs,metadata,device,reduction='none')
                  cands = get_candidate(outputs,metadata,loss_multi,k=int(len(rgb_img)*remove_rate*2))
                  outputs_cands.update(cands)
                  print(list(outputs_cands.keys()))
                  loss_multi = {k: v.mean() for k, v in loss_multi.items()}
                else:
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

          #update noisy labels
          if phase == 'train' and noisy_phase == 'noise_removal':
            high_scores = dict(sorted(outputs_cands.items(),key=lambda t: t[1][1], reverse=True)[:int(len(dataset[phase])*remove_rate)])
            for path, value in high_scores.items():
              dataset['train'].metadatas_dict[path] = value[0]

      if noisy_phase == 'normal':
          scheduler.step()

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
  # オプティマイザの定義
  optimizer = optim.Adam(model.parameters(), lr=config.TRAIN.LR, weight_decay=config.TRAIN.WEIGHT_DECAY)
  scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)

  # warmup_scheduler = CosineLRScheduler(optimizer, t_initial=config.TRAIN.NUM_EPOCHS - 20, lr_min=1e-7, warmup_t=20, warmup_prefix=True)

  log_path = os.path.join('log',os.path.splitext(config.SAVE_PATH)[0] + '.txt')
  os.makedirs(os.path.dirname(log_path), exist_ok=True)
  logger = init_config.init_logger(os.path.dirname(log_path), os.path.basename(log_path))

  init_config.set_random_seed(config.TRAIN.SEED)
  logger.info(config.dump())
  dump_path = os.path.join('config',os.path.splitext(config.SAVE_PATH)[0] + '.yaml')
  os.makedirs(os.path.dirname(dump_path), exist_ok=True)
  with open(dump_path,'w') as f:
    f.write(config.dump()) #type: ignore
    
  train(config, model, optimizer, scheduler, device=device)


if __name__ == '__main__':
   main()