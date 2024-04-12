from PIL import Image
import torch
from torch import nn
import torch.nn.functional as F
from clipseg.processing_clipseg import CLIPSegProcessor
from clipseg.modeling_clipseg import CLIPSegForImageSegmentation
from transformers.tokenization_utils_base import BatchEncoding


OBJ_CLASS_NAMES = [
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor",
]
OBJ_BASE_CLASS_NAMES = [
    c for i, c in enumerate(OBJ_CLASS_NAMES) if c not in ["bird", "car", "dog", "sheep", "motorbike"]
]

PARTS_NAMES = ["aeroplane's body", "aeroplane's stern", "aeroplane's wing", "aeroplane's tail", "aeroplane's engine", "aeroplane's wheel", "bicycle's wheel", "bicycle's saddle", "bicycle's handlebar", "bicycle's chainwheel", "bicycle's headlight", 
               "bird's wing", "bird's tail", "bird's head", "bird's eye", "bird's beak", "bird's torso", "bird's neck", "bird's leg", "bird's foot", "bottle's body", "bottle's cap", "bus's wheel", "bus's headlight", "bus's front", "bus's side", "bus's back", 
               "bus's roof", "bus's mirror", "bus's license plate", "bus's door", "bus's window", "car's wheel", "car's headlight", "car's front", "car's side", "car's back", "car's roof", "car's mirror", "car's license plate", "car's door", "car's window", 
               "cat's tail", "cat's head", "cat's eye", "cat's torso", "cat's neck", "cat's leg", "cat's nose", "cat's paw", "cat's ear", "cow's tail", "cow's head", "cow's eye", "cow's torso", "cow's neck", "cow's leg", "cow's ear", "cow's muzzle", "cow's horn", 
               "dog's tail", "dog's head", "dog's eye", "dog's torso", "dog's neck", "dog's leg", "dog's nose", "dog's paw", "dog's ear", "dog's muzzle", "horse's tail", "horse's head", "horse's eye", "horse's torso", "horse's neck", "horse's leg", "horse's ear", 
               "horse's muzzle", "horse's hoof", "motorbike's wheel", "motorbike's saddle", "motorbike's handlebar", "motorbike's headlight", "person's head", "person's eye", "person's torso", "person's neck", "person's leg", "person's foot", "person's nose", 
               "person's ear", "person's eyebrow", "person's mouth", "person's hair", "person's lower arm", "person's upper arm", "person's hand","pottedplant's pot", "pottedplant's plant", 
               "sheep's tail", "sheep's head", "sheep's eye", "sheep's torso", "sheep's neck", "sheep's leg", "sheep's ear", "sheep's muzzle", "sheep's horn", "train's headlight", "train's head", "train's front", "train's side", "train's back", "train's roof", 
               "train's coach", "tvmonitor's screen"]

BASE_PARTS_NAMES = [
    c for i, c in enumerate(PARTS_NAMES) if c not in ["bird's wing", "bird's tail", "bird's head", "bird's eye", "bird's beak", "bird's torso", "bird's neck", "bird's leg", "bird's foot",
                                                      "car's wheel", "car's headlight", "car's front", "car's side", "car's back", "car's roof", "car's mirror", "car's license plate", "car's door", "car's window",
                                                      "dog's tail", "dog's head", "dog's eye", "dog's torso", "dog's neck", "dog's leg", "dog's nose", "dog's paw", "dog's ear", "dog's muzzle",
                                                      "sheep's tail", "sheep's head", "sheep's eye", "sheep's torso", "sheep's neck", "sheep's leg", "sheep's ear", "sheep's muzzle", "sheep's horn",
                                                      "motorbike's wheel", "motorbike's saddle", "motorbike's handlebar", "motorbike's headlight"]
]

obj_part_map = {PARTS_NAMES.index(c): i for i,c in enumerate(BASE_PARTS_NAMES)}

# bird_parts = ["bird's wing", "bird's tail", "bird's head", "bird's eye", "bird's beak", "bird's torso", "bird's neck", "bird's leg", "bird's foot"]
bird_parts =  ["birds' head",
               "bird's beak",
               "bird's tail",
               "bird's wing",
               "bird's leg",
               "bird's eye",
               "bird's torso"]

with open('concepts/CUB/concepts_unique.txt', 'r') as fp:
    unique_concepts = fp.read().splitlines()

class CLIPSeg(nn.Module):
    def __init__(self):
        super().__init__()
        self.clipseg_processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")

        self.part_texts = [p.replace('\'s', '') for p in bird_parts]
        self.concept_texts = []
        for cpt in unique_concepts:
            idx = cpt.index(' ')
            self.concept_texts.append(cpt[:idx] + ' bird' + cpt[idx:])
        
        self.clipseg_model = CLIPSegForImageSegmentation.from_pretrained(
            "CIDAS/clipseg-rd64-refined"
        )
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        for name, params in self.clipseg_model.named_parameters():
            if 'clip.text_model.embeddings' in name or 'film' in name or 'visual_adapter' in name or 'decoder' in name: # VA+L+F+D
                params.requires_grad = True
            else:
                params.requires_grad = False
        self.part_text_encoding =  self.clipseg_processor.tokenizer(self.part_texts, return_tensors="pt", padding="max_length").to(self.device)
        self.concept_text_encoding = self.clipseg_processor.tokenizer(self.concept_texts, return_tensors="pt", padding="max_length").to(self.device)

        self.to(self.device)
        
    def forward_features(
        self, model, images: torch.Tensor, text_encoding, device: torch.device
    ):
        image_inputs = self.clipseg_processor(images=images, return_tensors="pt").to(device)
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
        images = [im.to(self.device) for im in images]

        if not self.training:
            return self.inference(images=images)

        bs, num_parts, num_concepts = targets.shape

        part_outputs = self.forward_features(self.clipseg_model, images, self.part_text_encoding, self.device) # shape: [b,n,h,w]
        part_features = part_outputs.decoder_output.hidden_states[-1]  # shape: [b*(num_parts+1), num_tokens, reduce_dim]
        part_cls = part_features[:, 0, :].view(bs, num_parts+1, -1)[:, :-1, :] # shape: [bs, num_parts, reduce_dim]

        concept_outputs = self.forward_features(self.clipseg_model, images, self.concept_text_encoding, self.device) # shape: [b,n,h,w]
        concept_features = concept_outputs.decoder_output.hidden_states[-1]  # shape: [b*(num_concepts+1), num_tokens, reduce_dim]
        concept_cls = concept_features[:, 0, :].view(bs, num_concepts+1, -1)[:, :-1, :] # shape: [bs, num_concepts, reduce_dim]
        # Calculate cosine similarity
        part_cls_norm = F.normalize(part_cls, p=2, dim=-1)
        concept_cls_norm = F.normalize(concept_cls, p=2, dim=-1)
        cos_sim = torch.bmm(part_cls_norm, concept_cls_norm.permute(0, 2, 1))  # shape: [bs, num_parts, bs*num_concepts], range: [0, 1]
        
        loss = F.binary_cross_entropy_with_logits(cos_sim, targets)
        return loss

