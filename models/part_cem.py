import timm
import copy
import torch
from torch import nn
import torch.nn.functional as F
from torchvision.models import resnet101, ResNet
from clip.model import CLIP, ModifiedResNet, AttentionPool2d


class PartCEM(nn.Module):
    def __init__(self, backbone='resnet50', num_parts=7, num_classes=200, dropout=0.3) -> None:
        super(PartCEM, self).__init__()
        self.k = num_parts + 1
        self.backbone = timm.create_model(backbone, pretrained=True)
        self.dim = self.backbone.fc.weight.shape[-1]

        self.prototypes = nn.Parameter(torch.randn(self.k, self.dim))
        self.modulations = torch.nn.Parameter(torch.ones((1, self.k, self.dim)))

        self.softmax2d = nn.Softmax2d()
        self.dropout = nn.Dropout1d(p=dropout)
        self.class_fc = nn.Linear(self.dim, num_classes)
    
    def forward(self, x):
        x = self.backbone.forward_features(x)
        b, c, h, w = x.shape

        b, c, h, w = x.shape
        x_flat = x.view(b, c, h*w).permute(0, 2, 1) # shape: [b,h*w,c]
        maps = torch.cdist(x_flat, self.prototypes, p=2) # shape: [b,h*w,k]
        maps = maps.permute(0, 2, 1).reshape(b, -1, h, w) # shape: [b,k,h,w]
        maps = self.softmax2d(-maps) # shape: [b,k,h,w]

        parts = torch.einsum('bkhw,bchw->bkchw', maps, x).mean((-1,-2)) # shape: [b,k,h,w], [b,c,h,w] -> [b,k,c]
        parts_modulated = parts * self.modulations # shape: [b,k,c]
        parts_modulated_dropped = self.dropout(parts_modulated) # shape: [b,k,c]
        class_logits = self.class_fc(parts_modulated_dropped) # shape: [b,k,|y|]

        return parts, maps, class_logits


class PartCEMTV(nn.Module):
    def __init__(self, backbone: ResNet, num_parts=8, num_classes=200, dropout=0.1) -> None:
        super(PartCEMTV, self).__init__()
        self.num_landmarks = num_parts
        self.k = num_parts + 1
        self.conv1 = backbone.conv1
        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

        self.dim = backbone.fc.in_features

        self.prototypes = nn.Parameter(torch.randn(self.k, self.dim))
        self.modulations = torch.nn.Parameter(torch.ones((1, self.k, self.dim)))

        self.softmax2d = nn.Softmax2d()
        self.dropout = nn.Dropout1d(p=dropout)
        self.class_fc = nn.Linear(self.dim, num_classes)
    
    def forward(self, x):
        # Pretrained ResNet part of the model
        x = self.conv1(x) # shape: [b, 64, h1, w1], e.g. h1=w1=112
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x) # shape: [b, 64, h2, w2], e.g. h2=w2=56
        x = self.layer1(x) # shape: [b, 256, h2, w2], e.g. h2=w2=56
        x = self.layer2(x) # shape: [b, 512, h3, w3], e.g. h2=w2=28
        l3 = self.layer3(x) # shape: [b, 1024, h3, w3], e.g. h2=w2=28
        x = self.layer4(l3) # shape: [b, 2048, h4, w4], e.g. h2=w2=7

        b, c, h, w = x.shape
        h, w = h*2, w*2
        x = torch.nn.functional.interpolate(x, size=(h, w), mode='bilinear') # shape: [b, 2048, h, w], e.g. h=w=14

        x_flat = x.view(b, c, h*w).permute(0, 2, 1) # shape: [b,h*w,c]
        maps = torch.cdist(x_flat, self.prototypes, p=2) # shape: [b,h*w,k]
        maps = maps.permute(0, 2, 1).reshape(b, -1, h, w) # shape: [b,k,h,w]
        maps = self.softmax2d(-maps) # shape: [b,k,h,w]

        parts = torch.einsum('bkhw,bchw->bkchw', maps, x).mean((-1,-2)) # shape: [b,k,h,w], [b,c,h,w] -> [b,k,c]
        parts_modulated = parts * self.modulations # shape: [b,k,c]
        parts_modulated_dropped = self.dropout(parts_modulated) # shape: [b,k,c]
        class_logits = self.class_fc(parts_modulated_dropped) # shape: [b,k,|y|]

        return parts, maps, class_logits


class PartCEMTVCpt(nn.Module):
    def __init__(self, backbone: ResNet, num_parts=8, num_classes=200, num_concepts=112, dropout=0.1) -> None:
        super().__init__()
        self.num_landmarks = num_parts
        self.k = num_parts + 1
        self.conv1 = backbone.conv1
        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

        self.dim = backbone.fc.in_features

        self.prototypes = nn.Parameter(torch.randn(self.k, self.dim))
        self.modulations = torch.nn.Parameter(torch.ones((1, self.k, self.dim)))

        self.softmax2d = nn.Softmax2d()
        self.dropout = nn.Dropout1d(p=dropout)
        self.class_fc = nn.Linear(self.dim, num_classes)
        self.concept_fc = nn.Linear(self.dim, num_concepts)
    
    def forward(self, x):
        # Pretrained ResNet part of the model
        x = self.conv1(x) # shape: [b, 64, h1, w1], e.g. h1=w1=112
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x) # shape: [b, 64, h2, w2], e.g. h2=w2=56
        x = self.layer1(x) # shape: [b, 256, h2, w2], e.g. h2=w2=56
        x = self.layer2(x) # shape: [b, 512, h3, w3], e.g. h2=w2=28
        l3 = self.layer3(x) # shape: [b, 1024, h3, w3], e.g. h2=w2=28
        x = self.layer4(l3) # shape: [b, 2048, h4, w4], e.g. h2=w2=7

        b, c, h, w = x.shape
        h, w = h*2, w*2
        x = torch.nn.functional.interpolate(x, size=(h, w), mode='bilinear') # shape: [b, 2048, h, w], e.g. h=w=14

        x_flat = x.view(b, c, h*w).permute(0, 2, 1) # shape: [b,h*w,c]
        maps = torch.cdist(x_flat, self.prototypes, p=2) # shape: [b,h*w,k]
        maps = maps.permute(0, 2, 1).reshape(b, -1, h, w) # shape: [b,k,h,w]
        maps = self.softmax2d(-maps) # shape: [b,k,h,w]

        parts = torch.einsum('bkhw,bchw->bkchw', maps, x).mean((-1,-2)) # shape: [b,k,h,w], [b,c,h,w] -> [b,k,c]
        parts_modulated = parts * self.modulations # shape: [b,k,c]
        parts_modulated_dropped = self.dropout(parts_modulated) # shape: [b,k,c]
        class_logits = self.class_fc(parts_modulated_dropped) # shape: [b,k,|y|]
        concept_logits = self.concept_fc(parts_modulated)

        return parts, maps, class_logits, concept_logits


class CLIPSpatial(nn.Module):
    def __init__(self, clip_model: CLIP):
        super().__init__()
        assert type(clip_model.visual) == ModifiedResNet
        assert type(clip_model.visual.attnpool) == AttentionPool2d
        self.visual_backbone = copy.deepcopy(clip_model.visual)
        self.visual_backbone.attnpool = nn.Identity()
        self.attnpool = copy.deepcopy(clip_model.visual.attnpool)

    def forward(self, x: torch.Tensor):
        x = self.visual_backbone(x)
        b, c, h, w = x.size()  # [BS, 2048, 7, 7]
        x = x.permute(0, 2, 3, 1)  # [BS, 7, 7, 2048]
        x = x.reshape(-1, c)  # [b*7*7, 2048]
        x = x[..., None, None]  # [b*7*7, 2048, 1, 1]
        x = x.expand(-1, -1, 7, 7)  # [b*7*7, 2048, 7, 7]
        x = self.attnpool(x)  # [b*7*7, 1024]
        x = x.reshape(b, h, w, -1)  # [b, 7, 7, 1024]
        x = x.permute(0, 3, 1, 2)  # [b, 1024, 7, 7]
        return x


class PartCEMClip(nn.Module):
    def __init__(self, backbone: nn.Module, prototypes: torch.Tensor, *, num_parts=8, num_classes=200, dropout=0.1) -> None:
        super().__init__()
        self.k = num_parts + 1
        self.backbone = backbone
        self.dim = 1024
        self.prototypes = nn.Parameter(prototypes.unsqueeze(0)) # shape: [1, k, dim]
        self.modulations = torch.nn.Parameter(torch.ones((1, self.k, self.dim)))

        self.softmax2d = nn.Softmax2d()
        self.dropout = nn.Dropout1d(p=dropout)
        self.class_fc = nn.Linear(self.dim, num_classes)
    
    def forward(self, x):
        # Pretrained ResNet part of the model
        x = self.backbone(x)

        b, c, h, w = x.shape
        # h, w = h*2, w*2
        # x = torch.nn.functional.interpolate(x, size=(h, w), mode='bilinear') # shape: [b, 2048, h, w], e.g. h=w=14

        # x_flat = x.view(b, c, h*w).permute(0, 2, 1) # shape: [b,h*w,c]
        # x_flat_norm = F.normalize(x_flat, p=2, dim=-1) # shape: [b,h*w,c]
        # proto_norm = F.normalize(self.prototypes, p=2, dim=-1) # shape: [1,k,c]

        # maps = torch.einsum('bnc,bkc->bnk', x_flat_norm, proto_norm.expand(b, -1, -1)) # shape: [b,h*w,k]
        # maps = maps.permute(0, 2, 1).reshape(b, -1, h, w) # shape: [b,k,h,w]
        # maps = self.softmax2d(maps) # shape: [b,k,h,w]

        x_flat = x.view(b, c, h*w).permute(0, 2, 1) # shape: [b,h*w,c]
        maps = torch.cdist(x_flat, self.prototypes, p=2) # shape: [b,h*w,k]
        maps = maps.permute(0, 2, 1).reshape(b, -1, h, w) # shape: [b,k,h,w]
        maps = self.softmax2d(-maps) # shape: [b,k,h,w]


        parts = torch.einsum('bkhw,bchw->bkchw', maps, x).mean((-1,-2)) # shape: [b,k,h,w], [b,c,h,w] -> [b,k,c]
        parts_modulated = parts * self.modulations # shape: [b,k,c]
        parts_modulated_dropped = self.dropout(parts_modulated) # shape: [b,k,c]
        class_logits = self.class_fc(parts_modulated_dropped) # shape: [b,k,|y|]

        return parts, maps, class_logits
