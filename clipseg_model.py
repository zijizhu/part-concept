from PIL import Image
import torch
from torch import nn
import torch.nn.functional as F
from clipseg.processing_clipseg import CLIPSegProcessor
from clipseg.modeling_clipseg import CLIPSegForImageSegmentation
from transformers.tokenization_utils_base import BatchEncoding


class CLIPSeg(nn.Module):
    def __init__(self, part_texts, concept_texts):
        super().__init__()
        self.clipseg_processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")

        self.part_texts = part_texts
        self.concept_texts = concept_texts
        
        self.clipseg_model = CLIPSegForImageSegmentation.from_pretrained(
            "CIDAS/clipseg-rd64-refined"
        )
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        for name, params in self.clipseg_model.named_parameters():
            if 'clip.text_model.embeddings' in name or 'film' in name or 'visual_adapter' in name or 'decoder' in name: # VA+L+F+D
                params.requires_grad = True
            else:
                params.requires_grad = False
        # self.part_text_encoding =  self.clipseg_processor.tokenizer(self.part_texts, return_tensors="pt", padding="max_length").to(self.device)
       # self.concept_text_encoding = self.clipseg_processor.tokenizer(self.concept_texts, return_tensors="pt", padding="max_length").to(self.device)

        self.text_encoding = self.clipseg_processor.tokenizer(self.part_texts + self.concept_texts, return_tensors="pt", padding="max_length").to(self.device)

        self.to(self.device)
        
    def forward_features(
        self, model, image_inputs: torch.Tensor, text_encoding,
    ):
        all_inputs = BatchEncoding(data=dict(
            **image_inputs,
            **text_encoding,
            output_hidden_states=torch.tensor(True),
            output_attentions=torch.tensor(True)
        ), tensor_type='pt')
        outputs = model(**all_inputs)
        return outputs
    
    def inference(self, images: torch.Tensor):
        bs, c, h, w = images.shape
        with torch.no_grad():
            outputs = self.forward_features(
                self.clipseg_model,
                images,
                self.part_text_encoding,
                self.device,
            )
            upscaled_logits = nn.functional.interpolate(
                outputs.logits[:,:-1,:,:],
                size=(h, w),
                mode="bilinear",
            )
        logits = torch.sigmoid(upscaled_logits).squeeze(0)
        return logits, outputs
    
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

    def forward(self, images: list[torch.Tensor], targets: torch.Tensor, weights: torch.Tensor):
        if not self.training:
            return self.inference(images=images)
        targets = targets.to(self.device)
        image_inputs = self.clipseg_processor(images=images, return_tensors="pt").to(self.device)
        bs, num_parts, num_concepts = targets.shape

        # all_text_encoding = torch.cat([self.text_encoding, self.concept_text_encoding])

        outputs = self.forward_features(self.clipseg_model, image_inputs, self.text_encoding) # shape: [b,n,h,w]
        features = outputs.decoder_output.hidden_states[-1]  # shape: [bs*(num_parts+num_concepts+1), num_tokens, reduce_dim]
        cls_tokens = features[:, 0, :].view(bs, num_parts+num_concepts+1, -1)[:, :-1, :] # shape: [bs, num_parts+num_concepts, reduce_dim]
        part_cls, concept_cls = cls_tokens[:, :num_parts, :], cls_tokens[:, num_parts:, :]

        # part_outputs = self.forward_features(self.clipseg_model, image_inputs, self.part_text_encoding, self.device) # shape: [b,n,h,w]
        # part_features = part_outputs.decoder_output.hidden_states[-1]  # shape: [b*(num_parts+1), num_tokens, reduce_dim]
        # part_cls = part_features[:, 0, :].view(bs, num_parts+1, -1)[:, :-1, :] # shape: [bs, num_parts, reduce_dim]

        # concept_outputs = self.forward_features(self.clipseg_model, images, self.concept_text_encoding, self.device) # shape: [b,n,h,w]
        # concept_features = concept_outputs.decoder_output.hidden_states[-1]  # shape: [b*(num_concepts+1), num_tokens, reduce_dim]
        # concept_cls = concept_features[:, 0, :].view(bs, num_concepts+1, -1)[:, :-1, :] # shape: [bs, num_concepts, reduce_dim]

        # Calculate cosine similarity
        part_cls_norm = F.normalize(part_cls, p=2, dim=-1)
        concept_cls_norm = F.normalize(concept_cls, p=2, dim=-1)
        cos_sim = torch.bmm(part_cls_norm, concept_cls_norm.permute(0, 2, 1))  # shape: [bs, num_parts, bs*num_concepts], range: [0, 1]
        
        loss = F.binary_cross_entropy_with_logits(cos_sim, targets, weight=weights)
        return loss

