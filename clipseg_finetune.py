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
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.models import resnet101, ResNet101_Weights
from torch.utils.tensorboard import SummaryWriter
from lightning import seed_everything
from data.cub_parts_dataset import CUBPartsDataset, collate_fn
from clipseg_model import CLIPSeg

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def train_epoch(model, dataloader: DataLoader, optimizer: torch.optim,
                writer: SummaryWriter, dataset_size: int, epoch: int,
                device: torch.device, logger: logging.Logger):

    running_loss = 0
    model.train()
    for batch in tqdm(dataloader):
        loss = model(batch)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        running_loss += loss.item() * len(batch[0])

        # TODO: calculate eval metrics
    
    # Log running losses

    loss_avg = running_loss / dataset_size
    writer.add_scalar(f'Loss/train/loss', loss_avg, epoch)
    logger.info(f'EPOCH {epoch} Train Loss: {loss_avg:.4f}')
    
    # TODO: log metrics
    # epoch_acc = running_corrects / dataset_size
    # writer.add_scalar(f'Accuracy/train', epoch_acc, epoch)
    # logger.info(f'EPOCH {epoch} Train Acc: {epoch_acc:.4f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PartCEM')
    parser.add_argument('--dataset_dir', type=str, required=True)
    parser.add_argument('--dataset', type=str, choices=['PASCUB'], required=True)

    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--epochs', default=20, type=int)
    parser.add_argument('--lr', default=2e-4, type=float)
    parser.add_argument('--batch_size', default=16, type=int)

    parser.add_argument('--eval', default=False, action='store_true')
    args = parser.parse_args()
    print(args)

    seed_everything(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    log_dir = os.path.join(f'{args.dataset}_runs', datetime.now().strftime('%Y-%m-%d_%H-%M'))
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    with open(os.path.join(log_dir, 'hparams.json'), 'w+') as fp:
        json.dump(vars(args), fp=fp, indent=4)

    writer = SummaryWriter(log_dir=log_dir)
    writer.add_text('Dataset', args.dataset)
    writer.add_text('Device', str(device))
    writer.add_text('Learning rate', str(args.lr))
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
    
    if args.dataset == 'PASCUB':
        dataset_train = CUBPartsDataset(dataset_dir=os.path.join(args.dataset_dir, args.dataset), split='train')
    else:
        raise NotImplementedError
    
    dataloader_train = DataLoader(dataset_train, collate_fn=collate_fn, batch_size=args.batch_size, shuffle=True, num_workers=4)
    # dataloader_val= DataLoader(dataset_val, batch_size=args.batch_size, shuffle=True, num_workers=4)
    # dataloader_test = DataLoader(dataset_test, batch_size=args.batch_size, shuffle=True, num_workers=4)
    
    model = CLIPSeg()
    state_dict = torch.load('checkpoints/clipseg_ft_VA_L_F_D_voc.pth', map_location='cpu')
    model.load_state_dict(state_dict['model'])
    
    model.to(device=device)
    print(summary(model))
    
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 5, 0.5)

    if args.eval:
        exit(0)

    logger.info('Start training...')
    for epoch in range(args.epochs):
        train_epoch(model=model, dataloader=dataloader_train, optimizer=optimizer,
                    writer=writer,dataset_size=len(dataset_train), epoch=epoch, device=device, logger=logger)
        # test_epoch(model=model, dataloader=dataloader_test, writer=writer,
        #            dataset_size=len(dataset_test), epoch=epoch, device=device)
        torch.save({k: v.cpu() for k, v in model.state_dict().items()},
                   os.path.join(log_dir, 'checkpoint.pt'))
        scheduler.step()


