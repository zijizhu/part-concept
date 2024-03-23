import timm
import torch
from torch import nn
import torch.nn.functional as F

class PartCEM(nn.Module):
    def __init__(self, backbone='resnet50', num_concepts=112, num_classes=200) -> None:
        super(PartCEM, self).__init__()
        self.backbone = timm.create_model(backbone, pretrained=True)
        self.dim = self.backbone.fc.weight.shape[-1]
        self.concepts = nn.Parameter(torch.randn(num_concepts + 1, self.dim))
        self.fc = nn.Linear(num_concepts, num_classes)
    
    def forward(self, x):
        x = self.backbone.forward_features(x)
        b, c, h, w = x.shape
        h, w = h*2, w*2
        x = F.interpolate(x, size=(h, w), mode='bilinear') # shape: [b, c, h, w], e.g. c=2048, h=w=14
        conv_weights = self.concepts[..., None, None] # shape: [num_concepts + 1, c, 1, 1]

        score_maps = F.sigmoid(F.conv2d(x, conv_weights)) # shape: [b, num_concepts, h, w]
        scores = F.sigmoid(score_maps.sum((-1, -2))) # shape: [b, num_concepts]
        preds = self.fc(scores[..., :-1]) # shape: [b, num_classes]

        concepts_expanded = self.concepts[..., None, None].expand(-1, -1, h, w) # shape: [num_concepts, c, h, w]
        reconstructed = torch.einsum('bkhw,kchw->bchw', score_maps, concepts_expanded) # shape: [b, c, h, w]

        x = x.view(b, c, h*w).permute(0, 2, 1) # shape: [b, h*w, c]
        reconstructed = reconstructed.view(b, c, h*w).permute(0, 2, 1) # shape: [b, h*w, c]

        recon_loss = torch.sum(torch.cdist(x, reconstructed, p=2)) # shape: [h*w, h*w]
        return scores, preds, recon_loss
