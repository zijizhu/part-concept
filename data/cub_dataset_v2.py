import os
import json
import torch
import spacy
import numpy as np
import pandas as pd
from PIL import Image
from collections import defaultdict
from torch.utils.data import Dataset
import torchvision.transforms.functional as F

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


class CUBDatasetV2(Dataset):
    def __init__(self, dataset_dir: str, concept_processed_path: str, parts_path: str, split='train') -> None:
        super().__init__()
        self.split = split
        self.dataset_dir = dataset_dir
        file_path_df = pd.read_csv(os.path.join(dataset_dir, 'CUB_200_2011', 'images.txt'),
                                    sep=' ', header=None, names=['image_id', 'file_path'])
        img_class_df = pd.read_csv(os.path.join(dataset_dir, 'CUB_200_2011', 'image_class_labels.txt'),
                                    sep=' ', header=None, names=['image_id', 'class_id'])
        train_test_split_df = pd.read_csv(os.path.join(dataset_dir, 'CUB_200_2011', 'train_test_split.txt'),
                                            sep=' ', header=None, names=['image_id', 'is_train'])

        class_name_df= pd.read_csv(os.path.join(dataset_dir, 'CUB_200_2011', 'classes.txt'), sep=' ', header=None, names=['class_id', 'class_name'])
        self.class_names = class_name_df['class_name'].str.replace('_', ' ').str.split('.').str[-1].to_list()

        main_df = (file_path_df
                   .merge(img_class_df, on='image_id')
                   .merge(train_test_split_df, on='image_id'))
        
        main_df['image_id'] -= 1
        main_df['class_id'] -= 1
        
        train_mask = main_df['is_train'] == 1
        val_mask = ~train_mask
        train_img_ids= main_df.loc[train_mask, 'image_id'].unique()
        val_img_ids = main_df.loc[val_mask, 'image_id'].unique()

        self.main_df = main_df.set_index('image_id')
        self.img_ids = {
            'train': train_img_ids,
            'val': val_img_ids
        }

        # Process generated concepts
        concept_sets = defaultdict(set)
        with open(concept_processed_path, 'rb') as fp:
            concepts_processed = json.load(fp=fp)

        # Add a noun to purely adjective concepts
        for class_name, concept_dict in concepts_processed.items():
            for part_name, concepts in concept_dict.items():
                concepts_with_part_name = attach_part_name(concepts, part_name)
                concept_dict[part_name] = concepts_with_part_name
                concept_sets[part_name].update(concepts_with_part_name)

        concept_sets_sorted = {k: sorted(list(v)) for k, v in concept_sets.items()}

        self.all_concepts = []
        for v in concept_sets_sorted.values():
            self.all_concepts += v

        num_concepts = sum(len(v) for v in concept_sets_sorted.values())
        self.concept_matrix = np.zeros((len(concepts_processed), len(concept_sets_sorted), num_concepts))

        with open(parts_path, 'r') as fp:
            self.all_parts = fp.read().splitlines()

        for class_idx, class_name in enumerate(self.class_names):
            class_concepts = concepts_processed[class_name]
            for part_idx, part_name in enumerate(self.all_parts):
                cpt_indices = [self.all_concepts.index(cpt) for cpt in class_concepts[part_name]]
                self.concept_matrix[class_idx, part_idx, cpt_indices] = 1

        self.weight_matrix = np.ones((len(self.all_parts), num_concepts))
        for part_idx, part_name in enumerate(self.all_parts):
            part_concepts = concept_sets_sorted[part_name]
            cpt_indices = [self.all_concepts.index(cpt) for cpt in part_concepts]
            self.weight_matrix[part_idx, cpt_indices] = 50

    def __len__(self):
        return len(self.img_ids[self.split])

    def __getitem__(self,idx):

        img_id = self.img_ids[self.split][idx]

        file_path, class_id, _ = self.main_df.iloc[img_id]

        image = Image.open(os.path.join(self.dataset_dir, 'CUB_200_2011', 'images', file_path)).convert('RGB')
        part_cpt_mat = self.concept_matrix[class_id]
        return F.pil_to_tensor(image), torch.tensor(part_cpt_mat, dtype=torch.float32), torch.tensor(self.weight_matrix, dtype=torch.float32)


def collate_fn(batch):
    image_list, cpt_mat_list, weight_mat_list = list(zip(*batch))
    return image_list, torch.stack(cpt_mat_list), torch.stack(weight_mat_list)