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
from torch.utils.tensorboard import SummaryWriter

from clipseg_model import CLIPSeg
from data.cub_dataset_v2 import CUBDatasetV2, collate_fn

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def train_epoch(model, dataloader: DataLoader, optimizer: torch.optim,
                writer: SummaryWriter, dataset_size: int, epoch: int,
                device: torch.device, logger: logging.Logger):

    running_loss = 0
    model.train()
    for batch in tqdm(dataloader):
        images, concept_matrix, weight_matrix = batch
        concept_matrix, weight_matrix=concept_matrix.to(device), weight_matrix.to(device)
        loss = model(images, concept_matrix, weight_matrix)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        running_loss += loss.item() * len(images)
    
    # Log running losses
    loss_avg = running_loss / dataset_size
    writer.add_scalar(f'Loss/train/loss', loss_avg, epoch)
    logger.info(f'EPOCH {epoch} Train Loss: {loss_avg:.4f}')
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PartCEM')
    parser.add_argument('--dataset_dir', type=str, required=True)
    parser.add_argument('--dataset', type=str, choices=['CUB'], required=True)

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
    
    # image_preprocessor= CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
    if args.dataset == 'CUB':
        dataset_train = CUBDatasetV2(os.path.join(args.dataset_dir, 'CUB'),
                                     os.path.join('concepts', 'CUB', 'concepts_processed.json'),
                                     os.path.join('concepts', 'CUB', 'parts.txt'))
        dataloader_train = DataLoader(dataset=dataset_train, collate_fn=collate_fn, batch_size=16, shuffle=True)
    else:
        raise NotImplementedError
     
    part_texts = ['bird ' + part for part in dataset_train.all_parts]
    model= CLIPSeg(part_texts=part_texts, concept_texts=dataset_train.all_concepts)
    state_dict = torch.load('checkpoints/clipseg_pascub_ft.pt')
    model.load_state_dict(state_dict)
    
    print(summary(model))
    
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 5, 0.5)

    if args.eval:
        exit(0)
    
    model.train()

    logger.info('Start training...')
    for epoch in range(args.epochs):
        train_epoch(model=model, dataloader=dataloader_train, optimizer=optimizer,
                    writer=writer,dataset_size=len(dataset_train), epoch=epoch, device=device, logger=logger)
        torch.save({k: v.cpu() for k, v in model.state_dict().items()},
                   os.path.join(log_dir, 'checkpoint.pt'))
        scheduler.step()
