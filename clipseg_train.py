'''By Zhijie Zhu, UNSW'''

import os
import sys
import json
import torch
import spacy
import logging
import argparse
import numpy as np
import pickle as pkl
from tqdm import tqdm
from pathlib import Path
from torchinfo import summary
from datetime import datetime
from collections import defaultdict
from lightning import seed_everything
from torch.utils.data import DataLoader
from torchvision.datasets import StanfordCars
from torch.utils.tensorboard.writer import SummaryWriter

from clipseg_model import CLIPSeg
from data.cub_dataset_v2 import CUBDatasetSimple



def load_concepts_cars():
    concept_sets = defaultdict(set)
    with open('concepts/CARS/concepts_processed.json', 'rb') as fp:
        concepts_processed = json.load(fp=fp)

    # Add a noun to purely adjective concepts
    for class_name, concept_dict in concepts_processed.items():
        for part_name, concepts in concept_dict.items():
            concept_sets[part_name].update(concepts)

    concept_sets_sorted = {k: sorted(list(v)) for k, v in concept_sets.items()}
    all_concepts = set()
    for v in concept_sets_sorted.values():
        all_concepts.update(v)

    return concept_sets_sorted


def load_concepts_cub():
    nlp = spacy.load("en_core_web_sm")
    def attach_part_name(concepts: list[str], part_name: str):
        concepts_processed = []
        for cpt in concepts:
            doc = nlp(cpt)
            if not any('NOUN' == word.pos_ for word in doc):
                cpt = cpt + ' ' + part_name
            # if 'than' in cpt or 'male' in cpt:  # Note that this would cause Purple Finch to have 0 concept for torso and American GoldFinch to have 0 concept for head
            #     continue 
            concepts_processed.append(cpt)
        return concepts_processed

    concept_sets = defaultdict(set)
    with open('concepts/CUB/concepts_processed.json', 'rb') as fp:
        concepts_processed = json.load(fp=fp)

    # Add a noun to purely adjective concepts
    for class_name, concept_dict in concepts_processed.items():
        for part_name, concepts in concept_dict.items():
            concepts_with_part_name = attach_part_name(concepts, part_name)
            concept_dict[part_name] = concepts_with_part_name
            concept_sets[part_name].update(concepts_with_part_name)

    concept_sets_sorted = {k: sorted(list(v)) for k, v in concept_sets.items()}
    all_concepts = set()
    for v in concept_sets_sorted.values():
        all_concepts.update(v)

    return concept_sets_sorted

def train_epoch_stage1(model, dataloader: DataLoader, optimizer: torch.optim.Optimizer,
                       writer: SummaryWriter, dataset_size: int, epoch: int,
                       device: torch.device, logger: logging.Logger):
    running_ce_loss = 0
    running_proto_loss = 0
    running_total_loss = 0
    running_corrects = 0

    for batch in tqdm(dataloader):
        images, targets = batch
        targets = targets.to(device)
        ce_loss, proto_loss, logits = model(images, targets)

        total_loss = ce_loss + 2 * proto_loss
        total_loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        running_ce_loss += ce_loss.item() * len(images)
        running_proto_loss += proto_loss.item() * len(images)
        running_total_loss += total_loss.item() * len(images)
        running_corrects += torch.sum(torch.argmax(logits.data, dim=-1) == targets.data).item()

    # Log running losses
    ce_loss_avg = running_ce_loss / dataset_size
    proto_loss_avg = running_proto_loss / dataset_size
    total_loss_avg = running_total_loss / dataset_size
    epoch_acc = running_corrects / dataset_size

    writer.add_scalar(f'Loss/train/ce', ce_loss_avg, epoch)
    writer.add_scalar(f'Loss/train/proto', proto_loss_avg, epoch)
    writer.add_scalar(f'Loss/train/total', total_loss_avg, epoch)
    writer.add_scalar(f'Acc/train', epoch_acc, epoch)
    logger.info(f'EPOCH {epoch} CE Train Loss: {ce_loss_avg:.4f}')
    logger.info(f'EPOCH {epoch} Proto Train Loss: {proto_loss_avg:.4f}')
    logger.info(f'EPOCH {epoch} Total Train Loss: {total_loss_avg:.4f}')
    logger.info(f'EPOCH {epoch} Train Aac: {epoch_acc:.4f}')


def train_epoch_stage2(model, dataloader: DataLoader, optimizer: torch.optim.Optimizer,
                       writer: SummaryWriter, dataset_size: int, epoch: int,
                       device: torch.device, logger: logging.Logger):
    running_ce_loss = 0
    running_total_loss = 0
    running_corrects = 0

    for batch in tqdm(dataloader):
        images, targets = batch
        targets = targets.to(device)
        ce_loss, logits = model(images, targets)

        total_loss = ce_loss
        total_loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        running_ce_loss += ce_loss.item() * len(images)
        running_total_loss += total_loss.item() * len(images)
        running_corrects += torch.sum(torch.argmax(logits.data, dim=-1) == targets.data).item()

    # Log running losses
    ce_loss_avg = running_ce_loss / dataset_size
    total_loss_avg = running_total_loss / dataset_size
    epoch_acc = running_corrects / dataset_size

    writer.add_scalar(f'Loss/train/ce', ce_loss_avg, epoch)
    writer.add_scalar(f'Loss/train/total', total_loss_avg, epoch)
    writer.add_scalar(f'Acc/train', epoch_acc, epoch)
    logger.info(f'EPOCH {epoch} CE Train Loss: {ce_loss_avg:.4f}')
    logger.info(f'EPOCH {epoch} Total Train Loss: {total_loss_avg:.4f}')
    logger.info(f'EPOCH {epoch} Train Aac: {epoch_acc:.4f}')


@torch.no_grad() 
def val_epoch(model, dataloader: DataLoader, writer: SummaryWriter,
              dataset_size: int, epoch: int, device: torch.device, logger: logging.Logger):
    running_corrects = 0

    for batch in tqdm(dataloader):
        images, targets = batch
        targets = targets.to(device)
        logits = model(images, targets)[-1]

        running_corrects += torch.sum(torch.argmax(logits.data, dim=-1) == targets.data).item()

    # Log running losses
    epoch_acc = running_corrects / dataset_size

    writer.add_scalar(f'Acc/val', epoch_acc, epoch)
    logger.info(f'EPOCH {epoch} Val Aac: {epoch_acc:.4f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PartCEM')
    parser.add_argument('--dataset_dir', type=str, required=True)
    parser.add_argument('--dataset', type=str, choices=['CUB', 'CARS'], required=True)
    parser.add_argument('--layers', type=str, nargs='+', choices=['va', 'f', 'l', 'd'])

    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--epochs', default=15, type=int)
    parser.add_argument('--lr', default=2e-4, type=float)
    parser.add_argument('--batch_size', default=16, type=int)

    parser.add_argument('--no_search', action='store_true')
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
    
    if args.dataset == 'CUB':
        def collate_fn(batch):
            image_list, label_list = list(zip(*batch))
            return image_list, torch.stack(label_list)

        dataset_train = CUBDatasetSimple(os.path.join(args.dataset_dir, 'CUB'), split='train')
        dataset_val = CUBDatasetSimple(os.path.join(args.dataset_dir, 'CUB'), split='val')
        dataloader_train = DataLoader(dataset=dataset_train, collate_fn=collate_fn, batch_size=args.batch_size, shuffle=True)
        dataloader_val = DataLoader(dataset=dataset_val, collate_fn=collate_fn, batch_size=args.batch_size, shuffle=True)

        with open('concepts/CUB/parts.txt') as fp:
            part_texts = fp.read().splitlines()
        concept_sets = load_concepts_cub()
        meta_category_text = 'bird'
        state_dict = torch.load('checkpoints/clipseg_pascub_ft.pt')

    elif args.dataset == 'CARS':
        def collate_fn(batch):
            image_list, label_list = list(zip(*batch))
            return image_list, torch.tensor(label_list)

        dataset_train = StanfordCars(root=os.path.join(args.dataset_dir, 'CARS'), split='train', download=True)
        dataset_val = StanfordCars(root=os.path.join(args.dataset_dir, 'CARS'), split='test', download=True)
        dataloader_train = DataLoader(dataset=dataset_train, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
        dataloader_val = DataLoader(dataset=dataset_val, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)

        with open('concepts/CARS/parts.txt') as fp:
            part_texts = fp.read().splitlines()
        concept_sets = load_concepts_cars()
        meta_category_text = 'car'
        state_dict = torch.load('checkpoints/clipseg_ft_VA_L_F_D_voc.pth', map_location='cpu')
        state_dict = state_dict['model']
    else:
        raise NotImplementedError

    with open(os.path.join(log_dir, 'all_concepts.pkl'), 'wb') as fp:
        pkl.dump(concept_sets, file=fp)

    model = CLIPSeg(
        part_texts=part_texts,
        concepts_dict=concept_sets,
        ft_layers=ft_layers,
        meta_category_text=meta_category_text,
        k=50,
        state_dict=state_dict
    )
    
    print(summary(model))

    # Classification using prototypes
    logger.info('Start training stage 1...')
    optimizer = torch.optim.AdamW([{'params': model.clipseg_model.parameters()},
                                   {'params': model.prototypes, 'lr': args.lr * 100},
                                   {'params': model.proj.parameters(), 'lr': args.lr * 10},
                                   {'params': model.fc.parameters(), 'lr': args.lr * 10}], lr=args.lr)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 5, 0.5)

    model.train()
    for epoch in range(args.epochs):
        train_epoch_stage1(model=model, dataloader=dataloader_train, optimizer=optimizer,
                           writer=writer, dataset_size=len(dataset_train),
                           device=device, epoch=epoch, logger=logger)
        val_epoch(model=model, dataloader=dataloader_val, writer=writer, dataset_size=len(dataset_val),
                  device=device, epoch=epoch, logger=logger)
        torch.save({k: v.cpu() for k, v in model.state_dict().items()},
                   os.path.join(log_dir, 'checkpoint_stage1.pt'))
        scheduler.step()
    
    if args.no_search:
        exit(0)
    
    logger.info('Search concepts based on prototypes...')
    selected_idxs, affinities = model.search_concepts()
    np.savez(os.path.join(log_dir, 'selcted_concept_idxs'), **selected_idxs)
    np.savez(os.path.join(log_dir, 'weight_concept_affinities'), **affinities)

    logger.info('Start training stage 2...')
    optimizer = torch.optim.AdamW([{'params': model.clipseg_model.parameters()},
                                   {'params': model.prototypes, 'lr': args.lr * 100},
                                   {'params': model.proj.parameters(), 'lr': args.lr * 10},
                                   {'params': model.fc.parameters(), 'lr': args.lr * 10}], lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 5, 0.5)
    model.train()
    for epoch in range(args.epochs):
        train_epoch_stage2(model=model, dataloader=dataloader_train, optimizer=optimizer,
                           writer=writer, dataset_size=len(dataset_train),
                           device=device, epoch=epoch, logger=logger)
        val_epoch(model=model, dataloader=dataloader_val, writer=writer, dataset_size=len(dataset_val),
                  device=device, epoch=epoch, logger=logger)
        torch.save({k: v.cpu() for k, v in model.state_dict().items()},
                   os.path.join(log_dir, 'checkpoint_stage2.pt'))
        scheduler.step()
