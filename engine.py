import torch
import logging
from tqdm import tqdm
import torch.nn.functional as F
from collections import OrderedDict
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

log = logging.getLogger(__name__)

def train_epoch(model, dataloader: DataLoader, optimizer: torch.optim,
                writer: SummaryWriter, dataset_size: int, epoch: int):
    
    running_loss_dict = {'attribute': 0.0, 'label': 0.0,
                         'recon': 0.0, 'total': 0.0}
    running_corrects = 0

    for img_ids, imgs, labels, attrs in tqdm(dataloader):
        batch_size = img_ids.size(0)

        attr_preds, label_preds, recon_loss = model(imgs)

        attr_loss = F.cross_entropy(attr_preds, attrs)
        label_loss = F.cross_entropy(label_preds, labels)
        total_loss = attr_loss + label_loss + recon_loss

        loss_dict = {'attribute': attr_loss, 'label': label_loss,
                     'recon': recon_loss, 'total': total_loss}

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
    writer.add_scalar(f'Loss/train/accuracy', epoch_acc, epoch)
    log.info(f'EPOCH {epoch} Train Acc: {epoch_acc:.4f}')


def test_epoch(model, dataloader: DataLoader, writer: SummaryWriter,
               dataset_size: int, epoch: int):
    running_loss_dict = {'attribute': 0.0, 'label': 0.0,
                         'recon': 0.0, 'total': 0.0}
    running_corrects = 0

    for img_ids, imgs, labels, attrs in tqdm(dataloader):
        batch_size = img_ids.size(0)

        attr_preds, label_preds, recon_loss = model(imgs)

        attr_loss = F.cross_entropy(attr_preds, attrs)
        label_loss = F.cross_entropy(label_preds, labels)
        total_loss = attr_loss + label_loss + recon_loss

        loss_dict = {'attribute': attr_loss, 'label': label_loss,
                     'recon': recon_loss, 'total': total_loss}

        for k, v in loss_dict.items():
            running_loss_dict[k] += v * batch_size
        running_corrects += torch.sum(torch.argmax(label_preds.data, dim=-1) == labels.data).item()
    
    for loss_name, loss_val in loss_dict.items():
        running_loss_dict[loss_name] += loss_val / dataset_size
        writer.add_scalar(f'Loss/val/{loss_name}', loss_val, epoch)
        log.info(f'EPOCH {epoch} Val {loss_name.capitalize()} Loss: {loss_val:.4f}')

    epoch_acc = running_corrects / dataset_size
    writer.add_scalar(f'Loss/val/accuracy', epoch_acc, epoch)
    log.info(f'EPOCH {epoch} Val Acc: {epoch_acc:.4f}')
