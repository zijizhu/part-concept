import timm
import torch
from torch import nn
import torch.nn.functional as F


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
    

###################
# Older Variants
###################
    

class PartCEMV1(nn.Module):
    def __init__(self, backbone='resnet50', num_concepts=112, num_classes=200) -> None:
        super(PartCEMV1, self).__init__()
        self.k = num_concepts
        self.backbone = timm.create_model(backbone, pretrained=True)
        self.dim = self.backbone.fc.weight.shape[-1]
        self.concepts = nn.Parameter(torch.randn(self.k, self.dim))
        self.modulation = nn.Parameter(torch.randn(1, self.dim))
        self.background = nn.Parameter(torch.randn(1, self.dim))
        self.concept_fc = nn.Sequential(
            nn.Linear(self.dim, 1),
            nn.Sigmoid()
        )
        self.label_fc = nn.Linear(self.dim * self.k, num_classes)
    
    def forward(self, x):
        x = self.backbone.forward_features(x)
        b, c, h, w = x.shape

        x_flat = x.view(b, c, h*w).permute(0, 2, 1) # shape: [b, h*w, c]
        
        cpts_expanded = self.concepts[None, ...].expand(b, -1, -1) # shape: [b, k, c]
        bg_expanded = self.background[None, ...].expand(b, -1, -1) # shape: [b, 1, c]
        cpts_wbg = torch.cat([cpts_expanded, bg_expanded], dim=1) # shape: [b, k+1, c]
        
        dists = torch.cdist(cpts_wbg, x_flat, p=2) # shape: [b, k+1, h*w]
        dists = F.softmax(-dists, dim=1) # shape: [b, k+1, h*w]
        cpt_score_maps = dists.view(b, -1, h, w) # shape: [b, k+1, h, w]

        cpt_reprs = torch.einsum('bkn,bnc->bkc', dists[:, :-1, :], x_flat) # shape: [b, k, c]
        cpt_logits = self.concept_fc(cpt_reprs).squeeze(-1) # shape: [b, k]

        cpt_logits_expanded = cpt_logits[..., None]  # shape: [b, k, 1]
        mod_expanded = self.modulation[None, ...].expand(b, self.k, -1) # shape: [b, k, c]
        cpts_mixed = (cpt_logits_expanded * cpts_expanded +
                      (1 - cpt_logits_expanded) * mod_expanded) # shape: [b, k, c]
        label_logits = self.label_fc(cpts_mixed.view(b, -1)) # shape: [b, k*c] -> [b, |y|]

        return cpt_score_maps, cpt_logits, label_logits


class PartCEMV0(nn.Module):
    def __init__(self, backbone='resnet50', num_concepts=112, num_classes=200) -> None:
        super(PartCEMV0, self).__init__()
        self.backbone = timm.create_model(backbone, pretrained=True)
        self.dim = self.backbone.fc.weight.shape[-1]
        self.concepts = nn.Parameter(torch.randn(num_concepts + 1, self.dim))
        # self.fc = nn.Linear(num_concepts, num_classes)
        self.concept_fc = nn.Linear(self.dim, 1)
        self.sig = nn.Sigmoid()
        self.label_fc = nn.Linear(self.dim, num_classes)
    
    def forward(self, x):
        x = self.backbone.forward_features(x)
        b, c, h, w = x.shape
        h, w = h*2, w*2
        x = F.interpolate(x, size=(h, w), mode='bilinear') # shape: [b, c, h, w], e.g. c=2048, h=w=14

        x_expanded = x.view(b, c, h*w).permute(0, 2, 1) # shape: [b, h*w, c]
        concepts_expanded = self.concepts[None, ...].expand(b, -1, -1) # shape: [b, num_concepts + 1, c]
        dists = torch.cdist(concepts_expanded, x_expanded, p=2) # shape: [b, num_concepts + 1, h*w]
        # dists = self.sig(1 - dists) # shape: [b, num_concepts + 1, h*w]
        dists = F.softmax(-dists, dim=1) # shape: [b, num_concepts + 1, h*w]

        score_maps = dists.view(b, -1, h, w)
        concept_vecs = torch.einsum('bkn,bnc->bkc', dists[:, :-1, :], x_expanded) # shape: [b, num_concepts, c]
        concept_scores = self.sig(self.concept_fc(concept_vecs).squeeze(-1)) # shape: [b, num_concepts]
        preds = self.label_fc(concept_scores @ self.concepts[:-1, :])

        return score_maps, concept_scores, preds
