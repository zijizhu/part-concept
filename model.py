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
        # self.fc = nn.Linear(num_concepts, num_classes)
        self.concept_fc = nn.Linear(28*28, num_concepts)
        self.sig = nn.Sigmoid()
        self.fc = nn.Linear(self.dim, num_classes)
    
    def forward(self, x):
        x = self.backbone.forward_features(x)
        b, c, h, w = x.shape
        h, w = h*2, w*2
        x = F.interpolate(x, size=(h, w), mode='bilinear') # shape: [b, c, h, w], e.g. c=2048, h=w=14
        conv_weights = self.concepts[..., None, None] # shape: [num_concepts + 1, c, 1, 1]

        x_norm = x / torch.linalg.vector_norm(x, ord=2, dim=1, keepdim=True)
        conv_weights = conv_weights / torch.linalg.vector_norm(conv_weights, ord=2, dim=1, keepdim=True)
        # score_maps = F.sigmoid(F.conv2d(x_norm, conv_weights)) # shape: [b, num_concepts + 1, h, w]
        score_maps = F.conv2d(x_norm, conv_weights) # shape: [b, num_concepts + 1, h, w]
        # scores = F.sigmoid(score_maps[:, :-1, ...].sum((-1, -2))) # shape: [b, num_concepts]
        score_maps_flatten = score_maps[:, :-1, ...].view(b, -1, h*w)
        scores = self.sig(self.concept_fc(score_maps_flatten)).squeeze(-1)

        concepts_expanded = self.concepts[..., None, None].expand(-1, -1, h, w) # shape: [num_concepts + 1, c, h, w]
        reconstructed = torch.einsum('bkhw,kchw->bchw', score_maps, concepts_expanded) # shape: [b, c, h, w]

        reconstructed_pooled = self.backbone.global_pool(reconstructed) # shape: [b, c]
        preds = self.fc(reconstructed_pooled) # shape: [b, num_classes]

        x = x.view(b, c, h*w).permute(0, 2, 1) # shape: [b, h*w, c]
        reconstructed = reconstructed.view(b, c, h*w).permute(0, 2, 1) # shape: [b, h*w, c]

        recon_loss = torch.mean(torch.cdist(x.detach(), reconstructed, p=2)) # shape: [h*w, h*w] -> []
        recon_commit_loss = torch.mean(torch.cdist(x, reconstructed.detach(), p=2)) # shape: [h*w, h*w] -> []
        return score_maps, scores, preds, recon_loss + recon_commit_loss
