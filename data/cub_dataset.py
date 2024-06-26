import json
import os
import torch
import numpy as np
import pandas as pd
from PIL import Image
from typing import Optional
import torchvision.transforms as T
from torch.utils.data import Dataset
import torchvision.transforms.functional as F
from sklearn.model_selection import train_test_split

# Replace text of 112 relvant concepts with the following dict
replacement = {
    'breast pattern': 'torso',
    'belly pattern': 'torso',
    'back pattern': 'torso',
    'underparts': 'torso',
    'upperparts': 'torso',
    'breast': 'torso',
    'belly': 'torso',
    'back': 'torso',
    'body': 'torso',
    'forehead': 'head',
    'throat': 'head',
    'crown': 'head',
    'nape': 'head',
    'head pattern': 'head',
    'under tail': 'tail',
    'upper tail': 'tail',
    'tail pattern': 'tail',
    'wing pattern': 'wing',

    'rounded-wings': 'rounded',
    'pointed-wings': 'pointed',
    'beak length about the same as head': 'long beak',
    'beak length shorter than head': 'short beak'
}

DEFAULT_ATTR_INDICES = [1, 4, 6, 7, 10, 14, 15, 20, 21, 23, 25, 29, 30,
                        35, 36, 38, 40, 44, 45, 50, 51, 53, 54, 56, 57,
                        59, 63, 64, 69, 70, 72, 75, 80, 84, 90, 91, 93,
                        99, 101, 106, 110, 111, 116, 117, 119, 125, 126,
                        131, 132, 134, 145, 149, 151, 152, 153, 157, 158,
                        163, 164, 168, 172, 178, 179, 181, 183, 187, 188,
                        193, 194, 196, 198, 202, 203, 208, 209, 211, 212,
                        213, 218, 220, 221, 225, 235, 236, 238, 239, 240,
                        242, 243, 244, 249, 253, 254, 259, 260, 262, 268,
                        274, 277, 283, 289, 292, 293, 294, 298, 299, 304,
                        305, 308, 309, 310, 311]

bird_parts =  ["head",
               "beak",
               "tail",
               "wing",
               "leg",
               "eye",
               "torso"]

with open('concepts/CUB/concepts_v4.txt', 'r') as fp:
    original_concepts = fp.read().splitlines()
with open('concepts/CUB/concepts_unique.txt', 'r') as fp:
    unique_concepts = fp.read().splitlines()
with open('concepts/CUB/hierarchy.json') as fp:
    hierarchy = json.load(fp=fp)

original2unique = {}
for i, original_cpt in enumerate(original_concepts):
    original2unique[i] = unique_concepts.index(original_cpt)

part_concept_association = np.zeros((len(bird_parts), len(unique_concepts)), dtype=int)
for cpt_idx, cpt in enumerate(unique_concepts):
    part = hierarchy[cpt]
    part_idx = bird_parts.index(part)
    part_concept_association[part_idx, cpt_idx] = 1


def get_transforms(image_size=448):
    train_transforms = T.Compose([
            T.Resize(size=image_size, antialias=True),
            T.RandomHorizontalFlip(),
            T.ColorJitter(0.1),
            T.RandomAffine(degrees=90, translate=(0.2, 0.2), scale=(0.8, 1.2)),
            T.CenterCrop(image_size),
            T.ToTensor(),
            # T.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
        ])
    test_transforms = T.Compose([
            T.Resize(size=image_size, antialias=True),
            T.CenterCrop(size=image_size),
            T.ToTensor(),
            # T.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
        ])
    return train_transforms, test_transforms


def build_datasets(dataset_dir: str,
                   attr_subset: str,
                   use_class_level_attr: bool,
                   transforms=None,
                   val_size: float=0.1):
    file_path_df = pd.read_csv(os.path.join(dataset_dir, 'CUB_200_2011', 'images.txt'),
                                   sep=' ', header=None, names=['image_id', 'file_path'])
    img_class_df = pd.read_csv(os.path.join(dataset_dir, 'CUB_200_2011', 'image_class_labels.txt'),
                                sep=' ', header=None, names=['image_id', 'class_id'])
    train_test_split_df = pd.read_csv(os.path.join(dataset_dir, 'CUB_200_2011', 'train_test_split.txt'),
                                        sep=' ', header=None, names=['image_id', 'is_train_val'])
    
    img_attr_df = pd.read_csv(os.path.join(dataset_dir, 'CUB_200_2011', 'attributes', 'image_attribute_labels.txt'),
                                sep=' ', usecols=[0,1,2,3], header=None,
                                names=['image_id', 'attribute_id', 'is_present', 'certainty_id'])
    
    main_df = (file_path_df
                .merge(img_class_df, on='image_id')
                .merge(train_test_split_df, on='image_id'))

    attr_df = img_attr_df.merge(img_class_df, on='image_id')

    # Convert all 1-indices to 0-indices
    main_df['image_id'] -= 1
    main_df['class_id'] -= 1
    attr_df['image_id'] -= 1
    attr_df['attribute_id'] -= 1
    attr_df['class_id'] -= 1

    # Calibrate <is_present> label of attributes based on certainties
    present_mask = attr_df['is_present'] == 1
    is_present_calibrated = attr_df.loc[present_mask, 'certainty_id'].map({1:0, 2:0.5, 3:0.75, 4:1})
    not_present_calibrated = attr_df.loc[~present_mask, 'certainty_id'].map({1:0, 2:0.5, 3:0.25, 4:0})

    def cat(df_list) -> pd.DataFrame:
        # For fixing https://github.com/microsoft/pylance-release/issues/5630
        return pd.concat(df_list)

    attr_df['is_present_calibrated'] = cat([is_present_calibrated, not_present_calibrated]).sort_index()

    # Split train, val, test sets
    train_val_mask = main_df['is_train_val'] == 1
    test_mask = ~train_val_mask
    train_val_img_ids = main_df.loc[train_val_mask, 'image_id'].unique()
    train_img_ids, val_img_ids = train_test_split(train_val_img_ids, test_size=val_size, shuffle=True)
    test_img_ids = main_df.loc[test_mask, 'image_id'].unique()

    # Decides which set of attributes to use
    # Use the same process as described in original CBM paper
    # To compute the ratio of samples per class having each attribute
    # The final result depends on the randomly split training set
    is_train_mask = attr_df['image_id'].isin(train_img_ids)
    visible_mask = (attr_df['is_present'] == 1) | (attr_df['certainty_id'] != 1)
    class_attr_ratios = pd.pivot_table(attr_df[is_train_mask & visible_mask],
                                       values='is_present',
                                       index=['class_id'],
                                       columns=['attribute_id'],
                                       aggfunc='mean')
    
    assert attr_subset in ['cbm', 'majority_10', 'all']        
    if attr_subset == 'cbm':
        attr_indices = DEFAULT_ATTR_INDICES
    elif attr_subset == 'generate':
        active_classes_per_attr = np.sum(class_attr_ratios >= 0.5, axis=0)
        attr_indices, = np.nonzero(active_classes_per_attr > 10) # np.nonzero returns a tuple of arrays
    else:
        attr_indices = attr_df['attribute_id'].unique()

    if use_class_level_attr:
        class_attrs_df = (class_attr_ratios >= 0.5).astype(int)[attr_indices]
    else:
        class_attrs_df = None
    
    use_attr_mask = attr_df['attribute_id'].isin(attr_indices)
    attr_df = attr_df[use_attr_mask]

    attr_df = attr_df.set_index('image_id')
    main_df = main_df.set_index('image_id')
    # attr_name_df = pd.read_csv(os.path.join(dataset_dir, 'attributes.txt'), sep=' ',
    #                            names=['attr_id', 'attr_name']).drop(columns=['attr_id'])

    # if use_transforms:
    #     transforms_train, transforms_test = get_transforms(448)
    # else:
    #     transforms_train, transforms_test = None, None

    dataset_train = CUBDataset(dataset_dir=dataset_dir, info_df=main_df, attributes_df=attr_df,
                               split_image_ids=train_img_ids, class_attributes_df=class_attrs_df,
                               transforms=transforms)
    dataset_val = CUBDataset(dataset_dir=dataset_dir, info_df=main_df, attributes_df=attr_df,
                             split_image_ids=val_img_ids, class_attributes_df=class_attrs_df,
                             transforms=transforms)
    dataset_test = CUBDataset(dataset_dir=dataset_dir, info_df=main_df, attributes_df=attr_df,
                              split_image_ids=test_img_ids, class_attributes_df=class_attrs_df,
                              transforms=transforms)
    return (dataset_train, dataset_val, dataset_test), attr_indices, class_attrs_df


class CUBDataset(Dataset):
    def __init__(self,
                 dataset_dir: str,
                 info_df: pd.DataFrame,
                 attributes_df: pd.DataFrame,
                 split_image_ids: np.ndarray,
                 class_attributes_df: Optional[pd.DataFrame]=None,
                 transforms=None) -> None:
        super().__init__()
        self.dataset_dir = dataset_dir
        self.info_df = info_df
        self.attr_df = attributes_df
        self.image_ids = split_image_ids
        self.class_attr_df = class_attributes_df
        self.transforms = transforms

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]

        file_path, class_id, _ = self.info_df.iloc[img_id]
        if self.class_attr_df is not None:
            attributes = self.class_attr_df.loc[class_id].to_numpy()
        else:
            attributes = self.attr_df.loc[img_id, 'is_present'].to_numpy()
        
        idxs, = np.where(attributes == 1)

        for k, v in original2unique.items():
            idxs[idxs == k] = v

        mapped_attr_labels = np.zeros(len(unique_concepts), dtype=int)
        mapped_attr_labels[idxs] = 1
        mapped_attr_labels_expanded = np.stack([mapped_attr_labels] * len(bird_parts))
        attr_labels_contrastive = mapped_attr_labels_expanded & part_concept_association

        image = Image.open(os.path.join(self.dataset_dir, 'CUB_200_2011', 'images', file_path)).convert('RGB')
        if self.transforms is not None:
            image = self.transforms(image)
        else:
            image = F.pil_to_tensor(image)
    
        return (torch.tensor(img_id),
                image,
                torch.tensor(class_id),
                torch.tensor(attr_labels_contrastive, dtype=torch.float32))

def collate_fn(batch):
    img_ids, images, class_ids, attr_labels = [], [], [], []
    for im_id, im, cls_id, attr_tgts in batch:
        img_ids.append(im_id)
        images.append(im)
        class_ids.append(cls_id)
        attr_labels.append(attr_tgts)
    return torch.stack(img_ids), images, torch.stack(class_ids), torch.stack(attr_labels)
