#%%
import re
import clip
import copy
import torch
from torch import nn
from PIL import Image
from typing import Union
from clip.model import CLIP, ModifiedResNet, AttentionPool2d

device = torch.device('cpu')
clip_model, img_transforms = clip.load('RN50', device=device)
#%%
class CLIPModified(nn.Module):
    def __init__(self, model: CLIP):
        super(CLIPModified, self).__init__()
        assert type(model.visual) == ModifiedResNet
        assert type(model.visual.attnpool) == AttentionPool2d

        image_model = copy.deepcopy(model.visual)
        image_model.attnpool = nn.Identity()
        self.model = image_model
        self.attnpool = copy.deepcopy(model.visual.attnpool)

    def forward(self, x: torch.Tensor):
        f = self.model(x) # raw features without attn_pool
        b, c, h, w = f.shape  # [BS, 2048, 7, 7]
        g = self.attnpool(f)  # [BS, 1024]
        x = f.permute(0, 2, 3, 1)  # [BS, 7, 7, 2048]
        x = x.reshape(-1, c)  # [BS*7*7, 2048]
        x = x[..., None, None]  # [BS*7*7, 2048, 1, 1]
        x = x.expand(-1, -1, 7, 7)  # [BS*7*7, 2048, 7, 7]
        x = self.attnpool(x)  # [BS*7*7, 1024]
        x = x.reshape(b, h, w, -1)  # [BS, 7, 7, 1024]
        x = x.permute(0, 3, 1, 2)  # [BS, 1024, 7, 7]
        return f, g, x
    
clip_modified = CLIPModified(clip_model)
#%%
x = torch.randn(8, 3, 224, 224)
f, g, x = clip_modified(x)
x = torch.randn(8, 3, 224, 224)
f.shape, g.shape, x.shape
# %%
import os
import matplotlib.pyplot as plt