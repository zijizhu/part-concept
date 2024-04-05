import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import os
import torch
import skimage
import numpy as np
import pandas as pd
from torch import nn
from torchinfo import summary
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.nn import BatchNorm2d, Softmax2d
from torchvision.models.resnet import ResNet
from torchvision.models.resnet import resnet50, resnet101

from datasets import build_datasets
from models.part_cem import PartCEM, PartCEMTV, PartCEMTVCpt

(dataset_train, dataset_val, dataset_test), attr_indices, class_attrs_df = build_datasets(
    dataset_dir=os.path.join('datasets', 'CUB'),
    attr_subset='cbm',
    use_class_level_attr=True,
    image_size=448
)

val_loader = DataLoader(dataset=dataset_val, batch_size=4, shuffle=True, num_workers=4)
val_loader_iter = iter(val_loader)

backbone = resnet101(weights=None)
net = PartCEMTVCpt(backbone=backbone, num_parts=8, num_classes=200, num_concepts=112)
state_dict = torch.load('CUB_runs/2024-04-01_23-19/checkpoint.pt', map_location='cpu')
attributes = torch.load('CUB_runs/2024-04-01_23-19/attributes.pkl')
net.load_state_dict(state_dict)


COLORS = [[0.75,0,0],[0,0.75,0],[0,0,0.75],[0.5,0.5,0],[0.5,0,0.5],[0,0.5,0.5],[0.75,0.25,0],[0.75,0,0.25],[0,0.75,0.25],
    [0.75,0,0],[0,0.75,0],[0,0,0.75],[0.5,0.5,0],[0.5,0,0.5],[0,0.5,0.5],[0.75,0.25,0],[0.75,0,0.25],[0,0.75,0.25],
    [0.75,0,0],[0,0.75,0],[0,0,0.75],[0.5,0.5,0],[0.5,0,0.5],[0,0.5,0.5],[0.75,0.25,0],[0.75,0,0.25],[0,0.75,0.25]]

def landmarks_to_rgb(maps):
    """
    Converts the attention maps to maps of colors
    Parameters
    ----------
    maps: Tensor, [number of parts, width_map, height_map]
        The attention maps to display

    Returns
    ----------
    rgb: Tensor, [width_map, height_map, 3]
        The color maps
    """
    n_concepts, spatial, spatial = maps.shape
    rgb = np.zeros((spatial, spatial, 3))
    for m in range(n_concepts):
        for c in range(3):
            rgb[:, :, c] += maps[m, :, :] * COLORS[m % 25][c]
    return rgb

def save_maps(X: torch.Tensor, maps: torch.Tensor, epoch: int, model_name: str, device: torch.device) -> None:
    """
    Plot images, attention maps and landmark centroids.
    Parameters
    ----------
    X: Tensor, [batch_size, 3, width_im, height_im]
        Input images on which to show the attention maps
    maps: Tensor, [batch_size, number of parts, width_map, height_map]
        The attention maps to display
    epoch: int
        The current epoch
    model_name: str
        The name of the model
    device: torch.device
        The device to use

    Returns
    -------
    """
    vis_size = (256, 256)
    batch_size, n_concepts, spatial, spatial = maps.shape
    grid_x, grid_y = torch.meshgrid(torch.arange(spatial), torch.arange(spatial))
    grid_x = grid_x[None, None, ...]
    grid_y = grid_y[None, None, ...]
    map_sums = maps.sum(3).sum(2).detach() # shape: [b, n_concepts]
    maps_x = grid_x * maps
    maps_y = grid_y * maps
    loc_x = maps_x.sum(3).sum(2) / map_sums
    loc_y = maps_y.sum(3).sum(2) / map_sums

    fig, axs = plt.subplots(2, 2)
    for i, ax in enumerate(axs.reshape(-1)):
        if i < maps.shape[0]:
            landmarks = landmarks_to_rgb(maps[i, :-1, ...].numpy()) # shape: [n_concepts, w, h] -> [w, h, 3]
            print(landmarks.shape, X[i, ...].permute(1, 2, 0).numpy().shape)
            # w, h, c
            img_with_landmarks = skimage.transform.resize(landmarks, vis_size) + skimage.transform.resize(X[i, ...].permute(1, 2, 0).numpy(), vis_size)
            ax.imshow(np.clip(img_with_landmarks, a_min=0, a_max=1))
            x_coords = loc_y[i, 0:-1] * 256 / spatial
            y_coords = loc_x[i, 0:-1] * 256 / spatial
            cols = COLORS[0: n_concepts - 1]
            n = np.arange(n_concepts)
            for xi, yi, col_i, mark in zip(x_coords, y_coords, cols, n):
                ax.scatter(xi, yi, color=col_i, marker=f'${mark}$')

    # plt.savefig(f'./results_{model_name}/{epoch}_{np.random.randint(0, 10)}')
    # plt.close()


device = 'cpu'
net.eval()

with torch.no_grad():
    img_ids, imgs, labels, attrs= next(val_loader_iter)
    _, maps, scores, cpt_logits = net(imgs)
    scores = scores.detach()
scores.shape, maps.shape

save_maps(imgs, maps, 0, 'cub_interpolate', 'cpu')

values, idxs = torch.topk(F.softmax(cpt_logits[2,2,:], dim=-1), 3)
print()

idxs.numpy(), values

attr_df = pd.read_csv('datasets/CUB/attributes.txt', delimiter=' ', header=None, names=['attr_id', 'name']).set_index('attr_id')
attr_df.loc[attributes['use_attribute_indices']].reset_index(drop=True).iloc[idxs.numpy()]