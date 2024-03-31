import os
import sys
import json
import torch
import logging
import argparse
from pathlib import Path
from torchinfo import summary
from datetime import datetime
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.models import resnet101, ResNet101_Weights
from torch.utils.tensorboard import SummaryWriter
from lightning import seed_everything
from datasets import build_datasets
from models.part_cem import PartCEM, PartCEMTV
from engine import train_epoch, test_epoch


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PartCEM')
    parser.add_argument('--dataset_dir', type=str, required=True)
    parser.add_argument('--dataset', type=str, choices=['CUB'], required=True)
    parser.add_argument('--attr_subset', type=str, choices=['cbm', 'majority_10', 'all'], required=True)
    parser.add_argument('--use_class_level_attr', action='store_true', required=True)

    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--num_parts', type=int, default=8)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--image_size', default=448, type=int)
    parser.add_argument('--backbone', default='resnet50', type=str)

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
    
    if args.dataset == 'CUB':
        (dataset_train, dataset_val, dataset_test), attr_indices, class_attrs_df = build_datasets(
            dataset_dir=os.path.join(args.dataset_dir, args.dataset),
            attr_subset=args.attr_subset,
            use_class_level_attr=args.use_class_level_attr,
            image_size=args.image_size
        )
        num_classes = 200
        torch.save({'use_attribute_indices': attr_indices, 'class_level_attributes': class_attrs_df},
                   f=os.path.join(log_dir, 'attributes.pkl'))
    else:
        raise NotImplementedError
    
    dataloader_train = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True, num_workers=4)
    dataloader_val= DataLoader(dataset_val, batch_size=args.batch_size, shuffle=True, num_workers=4)
    dataloader_test = DataLoader(dataset_test, batch_size=args.batch_size, shuffle=True, num_workers=4)
    
    # model = PartCEM(backbone=args.backbone,
    #                 num_parts=args.num_parts,
    #                 num_classes=num_classes)
    backbone = resnet101(ResNet101_Weights.DEFAULT)
    model = PartCEMTV(backbone=backbone,
                      num_parts=args.num_parts,
                      num_classes=num_classes)
    
    model.to(device=device)
    print(summary(model))

    high_lr_layers, med_lr_layers = ['modulations', 'prototypes'], ['class_fc']

    # First entry contains parameters with high lr, second with medium lr, third with low lr
    param_dicts = [{'params': [], 'lr': args.lr * 100},
                  {'params': [], 'lr': args.lr * 10},
                  {'params' : [], 'lr': args.lr}]
    for name, p in model.named_parameters():
        layer_name = name.split('.')[0]
        if layer_name in high_lr_layers:
            param_dicts[0]['params'].append(p)
        elif layer_name in med_lr_layers:
            param_dicts[1]['params'].append(p)
        else:
            param_dicts[2]['params'].append(p)
    
    optimizer = torch.optim.Adam(params=param_dicts)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 5, 0.5)

    if args.eval:
        exit(0)

    logger.info('Start training...')
    for epoch in range(args.epochs):
        train_epoch(model=model, dataloader=dataloader_train, optimizer=optimizer,
                    writer=writer,dataset_size=len(dataset_train), epoch=epoch, device=device)
        test_epoch(model=model, dataloader=dataloader_test, writer=writer,
                   dataset_size=len(dataset_test), epoch=epoch, device=device)
        torch.save({k: v.cpu() for k, v in model.state_dict().items()},
                   os.path.join(log_dir, 'checkpoint.pt'))
        scheduler.step()
