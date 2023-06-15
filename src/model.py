from einops import rearrange
import numpy as np
import timm
import torch
from torch import nn
import torch.nn.functional as f
from transformers import ResNetModel
from torchvision.ops import FeaturePyramidNetwork

class Regressor(nn.Module):
    def __init__(self, in_dim, hidden_dim) -> None:
        super(Regressor, self).__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            # nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            # nn.Dropout(0.3),
            # nn.Linear(hidden_dim, hidden_dim),
            # # nn.BatchNorm1d(hidden_dim),
            # nn.ReLU(),
            # nn.Dropout(0.3),
        )
        nn.Linear(hidden_dim,hidden_dim)
        self.regress = nn.ModuleDict({
            x: nn.Sequential(
                nn.Linear(hidden_dim,hidden_dim),
                # nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                # nn.Dropout(0.3),
                nn.Linear(hidden_dim,1),
            ) for x in ['cal','mass','fat','carb','protein']
        })

    def forward(self, x):
        x = self.fc1(x)
        out = {d: self.regress[d](x) for d in ['cal','mass','fat','carb','protein']}
        return out  
    
class SelfAttention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64):
        super().__init__()
        inner_dim = dim_head *  heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim = -1)

        self.to_q = nn.Linear(dim, inner_dim, bias = False)
        self.to_k = nn.Linear(dim, inner_dim, bias = False)
        self.to_v = nn.Linear(dim, inner_dim, bias = False)
        self.to_out = nn.Linear(inner_dim, dim, bias = False)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
        self.to_v.weight.data *= 0.67
        self.to_out.weight.data *= 0.67

    def forward(self, x):
        q = self.to_q(x)
        k = self.to_k(x)
        v = self.to_v(x)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), [q, k, v])

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        # out = out + x
        # out = self.norm(out + x)
        return out

class SimpleInceptionV2(nn.Module):
    def __init__(self, hidden_dim, pretrained_model: str = 'inception_resnet_v2') -> None:
        super(SimpleInceptionV2, self).__init__()
        self.conv0 = nn.Conv2d(4,3,1)
        self.backbone = timm.create_model(pretrained_model, features_only=True, pretrained=True)
        self.pooling = nn.AvgPool2d(3,2)
        self.flatten = nn.Flatten()
        self.regressor = Regressor(6144, hidden_dim)

    def forward(self, img, depth):
        x = torch.cat([img,depth],dim=1)
        x = self.conv0(x)
        x = self.backbone(x)[-1]
        x = self.pooling(x)
        x = self.flatten(x)
        out = self.regressor(x)
        return out  

class FPN(nn.Module):
    def __init__(self, hidden_dim: int, pretrained_model: str = 'microsoft/resnet-50', resolution_level: int = 2) -> None:
        super(FPN, self).__init__()
        self.backbone = nn.ModuleDict({x: ResNetModel.from_pretrained(pretrained_model) for x in ['rgb_img','depth_img']}) #type: ignore
        channel_list = [256, 512, 1024, 2048]
        self.resolution_level = resolution_level
        self.fpn = FeaturePyramidNetwork(channel_list,hidden_dim)
        self.attention = SelfAttention(hidden_dim)
        self.pooling = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()
        self.regressor = Regressor(hidden_dim * len(channel_list), hidden_dim * len(channel_list))
        for module in self.regressor.modules():
            if isinstance(module, nn.Linear):
                module.weight.data *= 0.67

    def forward(self, rgb_img: torch.Tensor, depth_img: torch.Tensor):
        B, _, _, _ = rgb_img.shape
        depth_img = depth_img.expand(-1,3,-1,-1)
        inputs = {'rgb_img': rgb_img, 'depth_img': depth_img}
        hidden_states_list = []
        for key, x in inputs.items():
            assert isinstance(self.backbone[key], ResNetModel)
            hidden_states = self.backbone[key](x,output_hidden_states=True).hidden_states
            hidden_states = hidden_states[1:]
            hidden_states_list.append(hidden_states)
        hidden_states_dict = {i: hid1 + hid2 for i, (hid1, hid2) in enumerate(zip(*hidden_states_list))}
        hidden_states = self.fpn(hidden_states_dict)
        fin_res = hidden_states[self.resolution_level].shape[-1]
        for i, hid in hidden_states.items():
            if i <= self.resolution_level:
                hidden_states[i] = f.adaptive_avg_pool2d(hid,fin_res)
            else:
                hidden_states[i] = f.interpolate(hid,fin_res)
        hidden_states = torch.stack([emb for emb in hidden_states.values()])
        pooled_hidden = hidden_states.mean(0)
        pooled_hidden = rearrange(pooled_hidden, 'b c h w -> b (h w) c', h=fin_res)
        pooled_hidden = self.attention(pooled_hidden)
        pooled_hidden = rearrange(pooled_hidden, 'b (h w) c -> b c h w', h=fin_res)
        hidden_states = hidden_states + pooled_hidden.unsqueeze(0)
        hidden_states = rearrange(hidden_states, 'n b c h w -> b (n c) h w', b=B)
        emb = self.pooling(hidden_states)
        emb = self.flatten(emb)
        out = self.regressor(emb)
        return out


def get_model(config,device):
    mod = config.MODEL.NAME
    pretrained_model = config.MODEL.PRETRAINED
    layers = config.TRAIN.LAYERS
    finetune = config.TRAIN.FINETUNE
    if mod == 'inceptionv2':
        model = SimpleInceptionV2(4096, pretrained_model)
    elif mod == 'resnet50':
        model = FPN(512,pretrained_model)
    elif mod == 'resnet101':
        model = FPN(512,pretrained_model)
    else:
        exit(1)

    # model.backbone.requires_grad_(False)
    return model
