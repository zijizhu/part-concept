import copy
import torch
import logging
from tqdm import tqdm
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from models.loss import conc_loss, orth_loss, pres_loss, landmark_coordinates

log = logging.getLogger(__name__)

loss_coefs = dict(
    label=2.0,
    conc=1000.0,
    orth=1.0,
    pres=1.0
)


def train_epoch(model, dataloader: DataLoader, optimizer: torch.optim,
                writer: SummaryWriter, dataset_size: int, epoch: int,
                device: torch.device):

    running_loss_dict = {k: 0.0 for k in loss_coefs}
    running_corrects = 0

    for img_ids, imgs, labels, attrs in tqdm(dataloader):
        imgs, labels, attrs = imgs.to(device), labels.to(device), attrs.to(device)
        batch_size = img_ids.size(0)

        parts, maps, preds = model(imgs)
        preds = preds[:, 0:-1, :].mean(1)

        # Calculate all losses
        cx, cy, grid_x, grid_y = landmark_coordinates(maps=maps, device=device)
        loss_dict = dict(
            label=F.cross_entropy(preds, labels),
            # conc=conc_loss(cx, cy, grid_x, grid_y, maps=maps),
            # orth=orth_loss(parts=parts, device=device),
            # pres=pres_loss(maps=maps)
        )

        # Calculate total Loss
        total_loss = torch.tensor(0.0, device=device)
        for k, v in loss_dict.items():
            total_loss += loss_coefs[k] * v

        total_loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        # Compute running losses and number of correct predictions
        for k, v in loss_dict.items():
            running_loss_dict[k] += v * batch_size
        running_corrects += torch.sum(torch.argmax(preds.data, dim=-1) == labels.data).item()
    
    # Log running losses
    for loss_name, loss_val in loss_dict.items():
        running_loss_dict[loss_name] += loss_val / dataset_size
        writer.add_scalar(f'Loss/train/{loss_name}', loss_val, epoch)
        log.info(f'EPOCH {epoch} Train {loss_name.capitalize()} Loss: {loss_val:.4f}')

    # Log accuracy
    epoch_acc = running_corrects / dataset_size
    writer.add_scalar(f'Accuracy/train', epoch_acc, epoch)
    log.info(f'EPOCH {epoch} Train Acc: {epoch_acc:.4f}')


@torch.no_grad()
def test_epoch(model, dataloader: DataLoader, writer: SummaryWriter,
               dataset_size: int, epoch: int, device: torch.device):

    running_loss_dict = {k: 0.0 for k in loss_coefs}
    running_corrects = 0

    for img_ids, imgs, labels, attrs in tqdm(dataloader):
        imgs, labels, attrs = imgs.to(device), labels.to(device), attrs.to(device)
        batch_size = img_ids.size(0)

        parts, maps, preds = model(imgs)
        preds = preds[:, 0:-1, :].mean(1)

        # Calculate all losses
        cx, cy, grid_x, grid_y = landmark_coordinates(maps=maps, device=device)
        loss_dict = dict(
            label=F.cross_entropy(preds, labels),
            # conc=conc_loss(cx, cy, grid_x, grid_y, maps=maps),
            # orth=orth_loss(parts=parts, device=device),
            # pres=pres_loss(maps=maps)
        )

        # Calculate total Loss
        total_loss = torch.tensor(0.0, device=device)
        for k, v in loss_dict.items():
            total_loss += loss_coefs[k] * v
        
        # Compute running losses and number of correct predictions
        for k, v in loss_dict.items():
            running_loss_dict[k] += v * batch_size
        running_corrects += torch.sum(torch.argmax(preds.data, dim=-1) == labels.data).item()
    
    for loss_name, loss_val in loss_dict.items():
        running_loss_dict[loss_name] += loss_val / dataset_size
        writer.add_scalar(f'Loss/val/{loss_name}', loss_val, epoch)
        log.info(f'EPOCH {epoch} Val {loss_name.capitalize()} Loss: {loss_val:.4f}')

    epoch_acc = running_corrects / dataset_size
    writer.add_scalar(f'Accuracy/val', epoch_acc, epoch)
    log.info(f'EPOCH {epoch} Val Acc: {epoch_acc:.4f}')
