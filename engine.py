import copy
import torch
import logging
from tqdm import tqdm
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from models.loss import conc_loss, orth_loss, pres_loss

log = logging.getLogger(__name__)

loss_dict = {'label': 0.0,
             'conc': 0.0,
             'orth': 0.0,
             'pres': 0.0}

def train_epoch(model, dataloader: DataLoader, optimizer: torch.optim,
                writer: SummaryWriter, dataset_size: int, epoch: int,
                device: torch.device):

    running_loss_dict = copy.deepcopy(loss_dict)
    running_corrects = 0

    for img_ids, imgs, labels, attrs in tqdm(dataloader):
        imgs, labels, attrs = imgs.to(device), labels.to(device), attrs.to(device)
        batch_size = img_ids.size(0)

        _, attr_preds, label_preds = model(imgs)
        attr_loss = F.binary_cross_entropy(attr_preds, attrs)
        label_loss = F.cross_entropy(label_preds, labels)
        total_loss = attr_loss + label_loss

        loss_dict = {'attribute': attr_loss, 'label': label_loss,
                     'total': total_loss}

        total_loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        for k, v in loss_dict.items():
            running_loss_dict[k] += v * batch_size
        running_corrects += torch.sum(torch.argmax(label_preds.data, dim=-1) == labels.data).item()
    
    for loss_name, loss_val in loss_dict.items():
        running_loss_dict[loss_name] += loss_val / dataset_size
        writer.add_scalar(f'Loss/train/{loss_name}', loss_val, epoch)
        log.info(f'EPOCH {epoch} Train {loss_name.capitalize()} Loss: {loss_val:.4f}')

    epoch_acc = running_corrects / dataset_size
    writer.add_scalar(f'Accuracy/train', epoch_acc, epoch)
    log.info(f'EPOCH {epoch} Train Acc: {epoch_acc:.4f}')


@torch.no_grad()
def test_epoch(model, dataloader: DataLoader, writer: SummaryWriter,
               dataset_size: int, epoch: int, device: torch.device):

    running_loss_dict = {'attribute': 0.0,
                         'label': 0.0,
                         'total': 0.0}
    running_corrects = 0

    for img_ids, imgs, labels, attrs in tqdm(dataloader):
        imgs, labels, attrs = imgs.to(device), labels.to(device), attrs.to(device)
        batch_size = img_ids.size(0)

        _, attr_preds, label_preds = model(imgs)

        attr_loss = F.binary_cross_entropy(attr_preds, attrs)
        label_loss = F.cross_entropy(label_preds, labels)
        total_loss = attr_loss + label_loss

        loss_dict = {'attribute': attr_loss, 'label': label_loss,
                     'total': total_loss}

        for k, v in loss_dict.items():
            running_loss_dict[k] += v * batch_size
        running_corrects += torch.sum(torch.argmax(label_preds.data, dim=-1) == labels.data).item()
    
    for loss_name, loss_val in loss_dict.items():
        running_loss_dict[loss_name] += loss_val / dataset_size
        writer.add_scalar(f'Loss/val/{loss_name}', loss_val, epoch)
        log.info(f'EPOCH {epoch} Val {loss_name.capitalize()} Loss: {loss_val:.4f}')

    epoch_acc = running_corrects / dataset_size
    writer.add_scalar(f'Accuracy/val', epoch_acc, epoch)
    log.info(f'EPOCH {epoch} Val Acc: {epoch_acc:.4f}')
