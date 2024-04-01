import torch
from torch import nn
import torch.nn.functional as F
import torchvision.transforms.functional as T


# def conc_loss(centroid_x: torch.Tensor, centroid_y: torch.Tensor, grid_x: torch.Tensor, grid_y: torch.Tensor,
#               maps: torch.Tensor) -> torch.Tensor:
#     """
#     Calculates the concentration loss, which is the weighted sum of the squared distance of the landmark
#     Parameters
#     ----------
#     centroid_x: torch.Tensor
#         The x coordinates of the map centroids
#     centroid_y: torch.Tensor
#         The y coordinates of the map centroids
#     grid_x: torch.Tensor
#         The x coordinates of the grid
#     grid_y: torch.Tensor
#         The y coordinates of the grid
#     maps: torch.Tensor
#         The attention maps

#     Returns
#     -------
#     loss_conc: torch.Tensor
#         The concentration loss
#     """
#     spatial_var_x = ((centroid_x.unsqueeze(-1).unsqueeze(-1) - grid_x) / grid_x.shape[-1]) ** 2
#     spatial_var_y = ((centroid_y.unsqueeze(-1).unsqueeze(-1) - grid_y) / grid_y.shape[-2]) ** 2
#     spatial_var_weighted = (spatial_var_x + spatial_var_y) * maps
#     loss_conc = spatial_var_weighted[:, 0:-1, :, :].mean()
#     return loss_conc


# def orth_loss(num_parts: int, landmark_features: torch.Tensor, device) -> torch.Tensor:
#     """
#     Calculates the orthogonality loss, which is the mean of the cosine similarities between every pair of landmarks
#     Parameters
#     ----------
#     num_parts: int
#         The number of landmarks
#     landmark_features: torch.Tensor, [batch_size, feature_dim, num_landmarks + 1 (background)]
#         Tensor containing the feature vector for each part
#     device: torch.device
#         The device to use
#     Returns
#     -------
#     loss_orth: torch.Tensor
#         The orthogonality loss
#     """
#     normed_feature = torch.nn.functional.normalize(landmark_features, dim=1)
#     similarity = torch.matmul(normed_feature.permute(0, 2, 1), normed_feature)
#     similarity = torch.sub(similarity, torch.eye(num_parts + 1).to(device))
#     loss_orth = torch.mean(torch.square(similarity))
#     return loss_orth


# def landmark_coordinates(maps: torch.Tensor, device: torch.device) -> \
#         tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
#     """
#     Calculate the coordinates of the landmarks from the attention maps
#     Parameters
#     ----------
#     maps: Tensor, [batch_size, number of parts, width_map, height_map]
#         Attention maps
#     device: torch.device
#         The device to use

#     Returns
#     -------
#     loc_x: Tensor, [batch_size, 0, number of parts]
#         The centroid x coordinates
#     loc_y: Tensor, [batch_size, 0, number of parts]
#         The centroid y coordinates
#     grid_x: Tensor, [batch_size, 0, width_map]
#         The x coordinates of the attention maps
#     grid_y: Tensor, [batch_size, 0, height_map]
#         The y coordinates of the attention maps
#     """
#     grid_x, grid_y = torch.meshgrid(torch.arange(maps.shape[2]),
#                                     torch.arange(maps.shape[3]))
#     grid_x = grid_x.unsqueeze(0).unsqueeze(0).to(device)
#     grid_y = grid_y.unsqueeze(0).unsqueeze(0).to(device)

#     map_sums = maps.sum(3).sum(2).detach()
#     maps_x = grid_x * maps
#     maps_y = grid_y * maps
#     loc_x = maps_x.sum(3).sum(2) / map_sums
#     loc_y = maps_y.sum(3).sum(2) / map_sums
#     return loc_x, loc_y, grid_x, grid_y


def pres_loss(maps: torch.Tensor):
    maps = maps[..., 2:-2, 2:-2]
    maps = F.avg_pool2d(maps, 3, stride=1)
    max_vals = F.max_pool2d(maps, maps.shape[-1]) # shape: [b, k, 1, 1]
    batch_pres, _ = max_vals.max(0) # shape: [k, 1, 1]
    return 1 - batch_pres.mean()


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
