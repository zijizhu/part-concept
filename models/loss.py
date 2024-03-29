import torch
from torch import nn
import torchvision.transforms.functional as F


def landmark_coordinates(maps: torch.Tensor, device: torch.device):
    """
    Calculate the coordinates of the landmarks from the attention maps
    Parameters
    ----------
    maps: Tensor, [batch_size, number of parts, width_map, height_map]
        Attention maps
    device: torch.device
        The device to use

    Returns
    -------
    loc_x: Tensor, [batch_size, 0, number of parts]
        The centroid x coordinates
    loc_y: Tensor, [batch_size, 0, number of parts]
        The centroid y coordinates
    grid_x: Tensor, [batch_size, 0, width_map]
        The x coordinates of the attention maps
    grid_y: Tensor, [batch_size, 0, height_map]
        The y coordinates of the attention maps
    """
    grid_x, grid_y = torch.meshgrid(torch.arange(maps.shape[2]),
                                    torch.arange(maps.shape[3]))
    grid_x = grid_x.unsqueeze(0).unsqueeze(0).to(device)
    grid_y = grid_y.unsqueeze(0).unsqueeze(0).to(device)

    map_sums = maps.sum(3).sum(2).detach()
    maps_x = grid_x * maps
    maps_y = grid_y * maps
    loc_x = maps_x.sum(3).sum(2) / map_sums
    loc_y = maps_y.sum(3).sum(2) / map_sums
    return loc_x, loc_y, grid_x, grid_y


def rigid_transform(img: torch.Tensor, angle: int, translate: list[int], scale: float, invert: bool=False):
    """
    Affine transforms input image
    Parameters
    ----------
    img: torch.Tensor
        Input image
    angle: int
        Rotation angle between -180 and 180 degrees
    translate: [int]
        Sequence of horizontal/vertical translations
    scale: float
        How to scale the image
    invert: bool
        Whether to invert the transformation

    Returns
    ----------
    img: torch.Tensor
        Transformed image
    """
    shear = 0
    bilinear = visionF.InterpolationMode.BILINEAR
    if not invert:
        img = visionF.affine(img, angle, translate, scale, shear,
                             interpolation=bilinear)
    else:
        translate = [-t for t in translate]
        img = visionF.affine(img, 0, translate, 1, shear)
        img = visionF.affine(img, -angle, [0, 0], 1/scale, shear)
    return img
