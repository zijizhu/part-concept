import torch
from torch import nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torchmetrics.functional import pairwise_cosine_similarity

from clipseg.processing_clipseg import CLIPSegProcessor
from clipseg.modeling_clipseg import CLIPSegForImageSegmentation
from transformers.tokenization_utils_base import BatchEncoding


class CLIPSeg(nn.Module):
    def __init__(self, part_texts, concepts_dict, meta_category_text, state_dict, ft_layers=[], k=50):
        super().__init__()
        self.k = k
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.clipseg_processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
        
        self.clipseg_model = CLIPSegForImageSegmentation.from_pretrained(
            "CIDAS/clipseg-rd64-refined"
        )
        
        for name, params in self.clipseg_model.named_parameters():
            # possible layers: 'clip.text_model.embeddings' 'film' 'visual_adapter' 'decoder' ie. VA, L, F, D
            if any(l in name for l in ft_layers):
                params.requires_grad = True
            else:
                params.requires_grad = False

        self.load_state_dict(state_dict)

        # Extra Parameters
        self.prototypes = nn.Parameter(torch.randn(len(part_texts), 512, self.k))
        self.proj = nn.Sequential(
            nn.Linear(512, 64, bias=False),
            nn.ReLU(inplace=True),
        )
        self.fc = nn.Linear(len(part_texts) * self.k, 200)

        self.to(self.device)

        self.part_texts = []
        self.concept_embedding_dict = dict() 
        with torch.no_grad():
            for p in part_texts:
                part_name = f'{meta_category_text} {p}'
                concepts_token = self.clipseg_processor.tokenizer(concepts_dict[p], return_tensors='pt', padding='max_length')
                part_concept_embeddings = self.clipseg_model.get_conditional_embeddings(**concepts_token.to(self.device))
                self.concept_embedding_dict[part_name] = part_concept_embeddings
                self.part_texts.append(part_name)
        self.all_concept_embeddings = torch.cat([emb for emb in self.concept_embedding_dict.values()])

        self.part_text_tokens = self.clipseg_processor.tokenizer(self.part_texts, return_tensors="pt", padding="max_length").to(self.device)

        self.register_buffer('selected_concept_embeddings', torch.zeros((len(self.part_texts), self.k, 512)))
        
    def forward_features(
        self, model, image_inputs: torch.Tensor, text_tokens,
    ):
        all_inputs = BatchEncoding(data=dict(
            **image_inputs,
            **text_tokens,
            output_hidden_states=torch.tensor(True),
            output_attentions=torch.tensor(True)
        ), tensor_type='pt')
        outputs = model(**all_inputs)
        return outputs

    @torch.no_grad()
    def inference(self, images: torch.Tensor):
        image_inputs = self.clipseg_processor(images=images, return_tensors="pt").to(self.device)
        bs, c, h, w = image_inputs['pixel_values'].shape

        outputs = self.forward_features(
            self.clipseg_model,
            image_inputs,
            self.part_text_tokens
        )
        upscaled_logits = nn.functional.interpolate(
            outputs.logits[:,:-1,:,:],
            size=(h, w),
            mode="bilinear",
        )

        logits = torch.sigmoid(upscaled_logits)
        return logits
    
    def segmentation_loss(self, part_outputs, tgt_list):
        '''tgt_list should include a channel for background'''
        logits = part_outputs.logits
        b, n, h, w = logits.shape  # n = num_classes+1 (background)

        # Interpolate all gt masks to the size of the model output
        dense_tgt = torch.stack(
          [F.interpolate(
              tgt[None, None, ...].float(),
              size=(h, w),
              mode="nearest"
          )
           for tgt
           in tgt_list]
        ).long()
        dense_tgt = dense_tgt.long().squeeze((1,2))  # shape: [b,h,w]

        one_hot_tgt = F.one_hot(dense_tgt, num_classes=n).float()  # {0,1}^[b,h,w,n]
        one_hot_tgt = one_hot_tgt.permute(0, 3, 1, 2)

        class_weight = torch.ones(n).to(self.device)
        class_weight[-1] = 0.05

        loss = F.binary_cross_entropy_with_logits(logits, one_hot_tgt, weight=class_weight[:, None, None])
        return loss

    def forward(self, images: list[torch.Tensor], targets: torch.Tensor):
        if not self.training:
            return self.inference(images=images)
        targets = targets.to(self.device)
        bs, = targets.shape
        image_inputs = self.clipseg_processor(images=images, return_tensors="pt").to(self.device)

        outputs = self.forward_features(self.clipseg_model, image_inputs, self.part_text_tokens) # shape: [b,n,h,w]
        features = outputs.decoder_output.hidden_states[-1]  # shape: [bs*(num_parts+1), num_tokens, reduce_dim]
        cls_tokens = features[:, 0, :].view(bs, len(self.part_texts) + 1, -1)[:, :-1, :] # shape: [bs, num_parts, reduce_dim]
        cls_tokens = cls_tokens.permute(1, 0, 2)  # shape: [num_parts, bs, reduce_dim]

        # Stage 1 forward:
        if torch.sum(self.selected_concept_embeddings) == 0:
            prototypes_projected = self.proj(self.prototypes.permute(0, 2, 1)).permute(0, 2, 1)
            prototype_logits = torch.bmm(cls_tokens, prototypes_projected).permute(1, 0, 2).contiguous()  # shape: [bs, num_parts, k]
            prototype_logits_flatten = prototype_logits.view(bs, len(self.part_texts) * self.k)  # shape: [bs, num_parts*k]
            class_logits = self.fc(prototype_logits_flatten)  # shape: [bs, num_classes]
        
            ce_loss = F.cross_entropy(class_logits, targets)

            # Calculate regularization loss
            prototypes_flat = self.prototypes.permute(0, 2, 1).reshape(len(self.part_texts) * self.k, 512)
            similarities = pairwise_cosine_similarity(prototypes_flat, self.all_concept_embeddings)
            prototype_loss = 1 - torch.mean(similarities)

            return ce_loss, prototype_loss, class_logits
        # Stage 2 forward
        else:
            concepts_projected = self.proj(self.selected_concept_embeddings).permute(0, 2, 1) # shape: [num_parts, reduce_dim, k]
            concept_logits = torch.bmm(cls_tokens, concepts_projected).permute(1, 0, 2).contiguous()  # shape: [bs, num_parts, k]
            concept_logits_flatten = concept_logits.view(bs, len(self.part_texts) * self.k)  # shape: [bs, num_parts*k]
            class_logits = self.fc(concept_logits_flatten)  # shape: [bs, num_classes]
            
            ce_loss = F.cross_entropy(class_logits, targets)
            return ce_loss, class_logits

    @torch.no_grad() 
    def search_concepts(self):
        weight_cpt_affinities, selected_cpt_idx_list, selected_cpt_embedding_list = [], [], []
        for part_name, part_prototypes in zip(self.part_texts, self.prototypes.permute(0, 2, 1).unbind(dim=0)):
            cpt_embeddings = self.concept_embedding_dict[part_name]
            affinities = pairwise_cosine_similarity(part_prototypes, cpt_embeddings).cpu().numpy()
            affinities = F.softmax(affinities)
            _, cpt_idxs = linear_sum_assignment(affinities)
            weight_cpt_affinities.append(affinities)
            selected_cpt_idx_list.append(cpt_idxs)
            selected_cpt_embedding_list.append(cpt_embeddings[cpt_idxs])
        # self.register_buffer('weight_cpt_affinities', torch.stack(weight_cpt_affinities))
        self.register_buffer('selected_concept_embeddings', torch.stack(selected_cpt_embedding_list))
