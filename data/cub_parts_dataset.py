import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset


class CUBPartsDataset(Dataset):
    def __init__(self, dataset_dir: str, split: str='train') -> None:
        super().__init__()
        self.split = split
        self.dataset_dir = dataset_dir
        pascal_filenames = os.listdir(os.path.join(dataset_dir, 'PASCAL_Parts', f'images_{split}'))
        cub_filenames = os.listdir(os.path.join(dataset_dir, 'CUB_Parts', f'images_{split}'))
        self.all_filenames = [('PASCAL_Parts', fn) for fn in pascal_filenames] + [('CUB_Parts', fn) for fn in cub_filenames]
        self.label2name = {
            0: 'background',
            1: 'head',
            2: 'beak/bill',
            3: 'tail',
            4: 'wing',
            5: 'leg',
            6: 'eye',
            7: 'torso'
        }

    def __len__(self):
        return len(self.all_filenames)
    
    def __getitem__(self, idx):
        subset, fn = self.all_filenames[idx]
        image = Image.open(os.path.join(self.dataset_dir,
                                        subset,
                                        f'images_{self.split}',
                                        fn)).convert('RGB')
        gt = Image.open(os.path.join(self.dataset_dir,
                                     subset,
                                     f'parts_{self.split}',
                                     fn))

        # Merge 4 and 5 (left and right wing), 8 and 9 (left and right eye)
        # And make label ids contiguous
        gt = torch.tensor(np.array(gt), dtype=torch.long)
        gt[gt == 5] = 4
        gt[gt == 6] = 5
        gt[gt == 7] = 5
        gt[gt == 8] = 6
        gt[gt == 9] = 6
        gt[gt == 10] = 7
        return image, gt.clone()
