import torch
import logging
import argparse
from tqdm import tqdm
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PartCEM')
    parser.add_argument('--model_name', help='Name under which the model will be saved', required=True)
    parser.add_argument('--data_root',
                    help='directory that contains the celeba, cub, or partimagenet folder', required=True)
    parser.add_argument('--dataset', help='The dataset to use. Choose celeba, cub, or partimagenet.', required=True)
    parser.add_argument('--num_parts', help='number of parts to predict', default=8, type=int)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--image_size', default=448, type=int) # 256 for celeba, 448 for cub,  224 for partimagenet
    parser.add_argument('--epochs', default=20, type=int) # 15 for celeba, 28 for cub, 20 for partimagenet
    parser.add_argument('--pretrained_model_path', default='', help='If you want to load a pretrained model,'
                        'specify the path to the model here.')
    parser.add_argument('--save_figures', default=False,
                        help='Whether to save the attention maps to png', action='store_true')
    parser.add_argument('--only_test', default=False, action='store_true', help='Whether to only test the model')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    writer = SummaryWriter(log_dir=f'{args.dataset}_runs/{args.model_name}')
    writer.add_text('Dataset', args.dataset.lower())
    writer.add_text('Device', str(device))
    writer.add_text('Learning rate', str(args.lr))
    writer.add_text('Batch size', str(args.batch_size))
    writer.add_text('Epochs', str(args.epochs))
    writer.add_text('Number of parts', str(args.num_parts))