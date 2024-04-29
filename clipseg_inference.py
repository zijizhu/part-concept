
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PartCEM')
    parser.add_argument('--dataset_dir', type=str, required=True)
    parser.add_argument('--dataset', type=str, choices=['CUB', 'CARS'], required=True)

    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--batch_size', default=16, type=int)

    args = parser.parse_args()
    print(args)

    seed_everything(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if args.dataset == 'CUB':
        def collate_fn(batch):
            image_list, label_list = list(zip(*batch))
            return image_list, torch.stack(label_list)

        dataset_train = CUBDatasetSimple(os.path.join(args.dataset_dir, 'CUB'), split='train')
        dataset_val = CUBDatasetSimple(os.path.join(args.dataset_dir, 'CUB'), split='val')
        dataloader_train = DataLoader(dataset=dataset_train, collate_fn=collate_fn, batch_size=args.batch_size, shuffle=False)
        dataloader_val = DataLoader(dataset=dataset_val, collate_fn=collate_fn, batch_size=args.batch_size, shuffle=False)

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
        dataloader_train = DataLoader(dataset=dataset_train, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
        dataloader_val = DataLoader(dataset=dataset_val, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

        with open('concepts/CARS/parts.txt') as fp:
            part_texts = fp.read().splitlines()
        concept_sets = load_concepts_cars()
        meta_category_text = 'car'
        state_dict = torch.load('checkpoints/clipseg_ft_VA_L_F_D_voc.pth', map_location='cpu')
        state_dict = state_dict['model']
    else:
        raise NotImplementedError

    model = CLIPSeg(
        part_texts=part_texts,
        concepts_dict=concept_sets,
        meta_category_text=meta_category_text,
        k=50,
        state_dict=state_dict
    )
    
    model.eval()

    results = []
    for batch in tqdm(dataloader_train):
        images, targets = batch
        targets = targets.to(device)
        logits = model(images, targets)
        results.append(logits)

    np.save(args.out_path, np.concatenate(results, axis=0))