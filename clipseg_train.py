'''By Zhijie Zhu, UNSW'''

import os
import sys
import json
import torch
import logging
import argparse
from tqdm import tqdm
from pathlib import Path
from torchinfo import summary
from datetime import datetime
from lightning import seed_everything
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter

from clipseg_model import CLIPSeg
from data.cub_dataset_v2 import CUBDatasetSimple


def train_epoch(model, dataloader: DataLoader, optimizer: torch.optim.Optimizer,
                writer: SummaryWriter, dataset_size: int, epoch: int,
                device: torch.device, logger: logging.Logger):
    running_loss = 0
    running_corrects = 0

    for batch in tqdm(dataloader):
        images, targets = batch
        targets = targets.to(device)
        loss, logits = model(images, targets)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        running_loss += loss.item() * len(images)
        running_corrects += torch.sum(torch.argmax(logits.data, dim=-1) == targets.data).item()

    # Log running losses
    loss_avg = running_loss / dataset_size
    epoch_acc = running_corrects / dataset_size

    writer.add_scalar(f'Loss/train', loss_avg, epoch)
    writer.add_scalar(f'Acc/train', epoch_acc, epoch)
    logger.info(f'EPOCH {epoch} Train Loss: {loss_avg:.4f}')
    logger.info(f'EPOCH {epoch} Train Aac: {epoch_acc:.4f}')


@torch.no_grad() 
def val_epoch(model, dataloader: DataLoader, writer: SummaryWriter,
              dataset_size: int, epoch: int, device: torch.device, logger: logging.Logger):
    running_corrects = 0

    for batch in tqdm(dataloader):
        images, targets = batch
        targets = targets.to(device)
        loss, logits = model(images, targets)

        running_corrects += torch.sum(torch.argmax(logits.data, dim=-1) == targets.data).item()

    # Log running losses
    epoch_acc = running_corrects / dataset_size

    writer.add_scalar(f'Acc/val', epoch_acc, epoch)
    logger.info(f'EPOCH {epoch} Val Aac: {epoch_acc:.4f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PartCEM')
    parser.add_argument('--dataset_dir', type=str, required=True)
    parser.add_argument('--dataset', type=str, choices=['CUB'], required=True)
    parser.add_argument('--layers', type=str, nargs='+', choices=['va', 'f', 'l', 'd'])

    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--epochs', default=30, type=int)
    parser.add_argument('--lr', default=2e-4, type=float)
    parser.add_argument('--batch_size', default=16, type=int)

    args = parser.parse_args()
    print(args)

    seed_everything(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ft_layer_dict = dict(
        va='visual_adapter',
        l='clip.text_model.embeddings',
        f='film',
        d='decoder'
    )
    ft_layers = [ft_layer_dict[l] for l in args.layers]

    log_dir = os.path.join(f'{args.dataset}_runs', datetime.now().strftime('%Y-%m-%d_%H-%M'))
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    with open(os.path.join(log_dir, 'hparams.json'), 'w+') as fp:
        json.dump(vars(args), fp=fp, indent=4)

    writer = SummaryWriter(log_dir=log_dir)
    writer.add_text('Dataset', args.dataset)
    writer.add_text('Device', str(device))
    writer.add_text('Learning rate', str(args.lr))
    writer.add_text('Fine-tune Layers', ', '.join(ft_layers))
    writer.add_text('Batch size', str(args.batch_size))
    writer.add_text('Epochs', str(args.epochs))
    writer.add_text('Seed', str(args.seed))

    logging.basicConfig(level=logging.INFO,
                        format='[%(asctime)s][%(name)s][%(levelname)s] - %(message)s', datefmt='%Y-%m-%d %H:%M:%S',
                        handlers=[
                            logging.FileHandler(os.path.join(log_dir, 'train.log')),
                            logging.StreamHandler(sys.stdout)
                        ])
    logger = logging.getLogger(__name__)
    
    def collate_fn(batch):
        image_list, label_list = list(zip(*batch))
        return image_list, torch.stack(label_list)

    if args.dataset == 'CUB':
        dataset_train = CUBDatasetSimple(os.path.join(args.dataset_dir, 'CUB'), split='train')
        dataset_val = CUBDatasetSimple(os.path.join(args.dataset_dir, 'CUB'), split='val')
        dataloader_train = DataLoader(dataset=dataset_train, collate_fn=collate_fn, batch_size=args.batch_size, shuffle=True)
        dataloader_val = DataLoader(dataset=dataset_val, collate_fn=collate_fn, batch_size=args.batch_size, shuffle=True)
    else:
        raise NotImplementedError

    with open('concepts/CUB/parts.txt') as fp:
        part_texts = ['bird ' + word for word in fp.read().splitlines()]
    
    state_dict = torch.load('checkpoints/clipseg_pascub_ft.pt')

    model = CLIPSeg(part_texts=part_texts, ft_layers=ft_layers, state_dict=state_dict)
    
    print(summary(model))
 
    optimizer = torch.optim.AdamW([{'params': model.clipseg_model.parameters()},
                                   {'params': model.prototypes, 'lr': args.lr * 10},
                                   {'params': model.proj.parameters(), 'lr': args.lr * 10},
                                   {'params': model.fc.parameters(), 'lr': args.lr * 10}], lr=args.lr)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 5, 0.5)

    model.train()

    logger.info('Start training...')
    for epoch in range(args.epochs):
        train_epoch(model=model, dataloader=dataloader_train, optimizer=optimizer,
                    writer=writer, dataset_size=len(dataset_train),
                    device=device, epoch=epoch, logger=logger)
        val_epoch(model=model, dataloader=dataloader_val, writer=writer, dataset_size=len(dataset_val),
                  device=device, epoch=epoch, logger=logger)
        torch.save({k: v.cpu() for k, v in model.state_dict().items()},
                   os.path.join(log_dir, 'checkpoint.pt'))
        scheduler.step()
