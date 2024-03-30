import torch
from torch import nn
import torch.nn.functional as F
import torchvision.transforms.functional as T


def pres_loss(maps: torch.Tensor):
    maps = maps[..., 2:-2, 2:-2]
    maps = F.avg_pool2d(maps, 3, stride=1)
    max_vals = F.max_pool2d(maps, maps.shape[-1]) # shape: [b, k, 1, 1]
    batch_pres, _ = max_vals.max(0) # shape: [k, 1, 1]
    return batch_pres.mean()


def conc_loss(centroid_x: torch.Tensor, centroid_y: torch.Tensor,
              grid_x: torch.Tensor, grid_y: torch.Tensor,
              maps: torch.Tensor) -> torch.Tensor:
    """
    Calculates the concentration loss, which is the weighted sum of the squared distance of the landmark
    Parameters
    ----------
    centroid_x: torch.Tensor
        The x coordinates of the map centroids
    centroid_y: torch.Tensor
        The y coordinates of the map centroids
    grid_x: torch.Tensor
        The x coordinates of the grid
    grid_y: torch.Tensor
        The y coordinates of the grid
    maps: torch.Tensor
        The attention maps

    Returns
    -------
    loss_conc: torch.Tensor
        The concentration loss
    """
    b, k, h, w = maps.shape
    spatial_var_x = ((centroid_x[..., None, None] - grid_x) / grid_x.shape[-1]) ** 2
    spatial_var_y = ((centroid_y[..., None, None] - grid_y) / grid_y.shape[-2]) ** 2
    spatial_var_weighted = (spatial_var_x + spatial_var_y) * maps
    loss_conc = spatial_var_weighted[:, :-1, ...].mean()
    return loss_conc


def orth_loss(parts: torch.Tensor, device) -> torch.Tensor:
    """
    Calculates the orthogonality loss, which is the mean of the cosine similarities between every pair of landmarks
    Parameters
    ----------
    num_parts: int
        The number of landmarks
    landmark_features: torch.Tensor, [batch_size, feature_dim, num_landmarks + 1 (background)]
        Tensor containing the feature vector for each part
    device: torch.device
        The device to use
    Returns
    -------
    loss_orth: torch.Tensor
        The orthogonality loss
    """
    b, k, c = parts.shape
    parts_norm = F.normalize(parts, p=2, dim=-1)
    similarity = torch.bmm(parts, parts_norm.transpose(1,2))
    similarity = torch.sub(similarity, torch.eye(k).to(device))
    loss_orth = torch.mean(torch.square(similarity))
    return loss_orth


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
    b, k, h, w = maps.shape
    grid_x, grid_y = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')
    grid_x = grid_x[None, None, ...].to(device) # shape: [1,1,h,w]
    grid_y = grid_y[None, None, ...].to(device) # shape: [1,1,h,w]

    maps_x = grid_x * maps
    maps_y = grid_y * maps

    map_sums = maps.sum((-1,-2)).detach() # shape: [b,k]
    centroid_x = maps_x.sum((-1,-2)) / map_sums # shape: [b,k]
    centroid_y = maps_y.sum((-1,-2)) / map_sums # shape: [b,k]
    return centroid_x, centroid_y, grid_x, grid_y


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
    bilinear = T.InterpolationMode.BILINEAR
    if not invert:
        img = T.affine(img, angle, translate, scale, shear,
                             interpolation=bilinear)
    else:
        translate = [-t for t in translate]
        img = T.affine(img, 0, translate, 1, shear)
        img = T.affine(img, -angle, [0, 0], 1/scale, shear)
    return img
