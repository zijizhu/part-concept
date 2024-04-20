import torch
from torch import nn
import torch.nn.functional as F
from clipseg.processing_clipseg import CLIPSegProcessor
from clipseg.modeling_clipseg import CLIPSegForImageSegmentation
from transformers.tokenization_utils_base import BatchEncoding


class CLIPSeg(nn.Module):
    def __init__(self, part_texts, state_dict):
        super().__init__()
        self.clipseg_processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")

        self.part_texts = part_texts
        
        self.clipseg_model = CLIPSegForImageSegmentation.from_pretrained(
            "CIDAS/clipseg-rd64-refined"
        )
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        for name, params in self.clipseg_model.named_parameters():
            # if 'clip.text_model.embeddings' in name or 'film' in name or 'visual_adapter' in name or 'decoder' in name: # VA+L+F+D
            #     params.requires_grad = True
            if 'film' in name or 'visual_adapter' in name or 'decoder' in name: # VA+L+F+D
                params.requires_grad = True
            else:
                params.requires_grad = False

        self.text_encoding = self.clipseg_processor.tokenizer(self.part_texts, return_tensors="pt", padding="max_length").to(self.device)

        self.load_state_dict(state_dict)

        # Two stage experiment
        self.prototypes = nn.Parameter(torch.randn(len(self.part_texts), 512, 50))
        self.proj = nn.Sequential(
            nn.Linear(512, 64, bias=False),
            nn.ReLU(inplace=True),
        )
        self.fc = nn.Linear(len(self.part_texts) * 50, 200)

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

    def forward(self, images: list[torch.Tensor], targets: torch.Tensor):
        if not self.training:
            return self.inference(images=images)
        targets = targets.to(self.device)
        bs, = targets.shape
        image_inputs = self.clipseg_processor(images=images, return_tensors="pt").to(self.device)

        outputs = self.forward_features(self.clipseg_model, image_inputs, self.text_encoding) # shape: [b,n,h,w]
        features = outputs.decoder_output.hidden_states[-1]  # shape: [bs*(num_parts+1), num_tokens, reduce_dim]
        cls_tokens = features[:, 0, :].view(bs, len(self.part_texts) + 1, -1)[:, :-1, :] # shape: [bs, num_parts, reduce_dim]

        cls_tokens = cls_tokens.permute(1, 0, 2)  # shape: [num_parts, bs, reduce_dim]
        prototypes_projected = self.proj(self.prototypes.permute(0, 2, 1)).permute(0, 2, 1)
        concept_logits = torch.bmm(cls_tokens, prototypes_projected).permute(1, 0, 2).contiguous()  # shape: [bs, num_parts, 5]
        concept_logits_flatten = concept_logits.view(bs, len(self.part_texts) * 50)  # shape: [bs, num_parts*5]
        class_logits = self.fc(concept_logits_flatten)  # shape: [bs, num_classes]
        
        loss = F.cross_entropy(class_logits, targets)
        return loss, class_logits
