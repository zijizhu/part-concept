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

class CLIPSeg(nn.Module):
    def __init__(self):
        super().__init__()
        self.clipseg_processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
        # train_class_texts = PARTS_NAMES
        # self.train_class_texts = [c.replace('\'s', '') for c in train_class_texts]
        # self.train_obj_classes = OBJ_CLASS_NAMES

        # self.test_class_texts = PARTS_NAMES
        # self.test_obj_classes = OBJ_CLASS_NAMES

        self.part_texts = [p.replace('\'s', '') for p in bird_parts]
        
        self.clipseg_model = CLIPSegForImageSegmentation.from_pretrained(
            "CIDAS/clipseg-rd64-refined"
        )
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        for name, params in self.clipseg_model.named_parameters():
            if 'clip.text_model.embeddings' in name or 'film' in name or 'visual_adapter' in name or 'decoder' in name: # VA+L+F+D
                params.requires_grad = True
            else:
                params.requires_grad = False
        # self.train_text_encoding = self.clipseg_processor.tokenizer(self.train_class_texts, return_tensors="pt", padding="max_length")
        self.part_text_encoding =  self.clipseg_processor.tokenizer(self.part_texts, return_tensors="pt", padding="max_length").to(self.device)
        
    def preds_to_semantic_inds(self, preds, threshold):
        flat_preds = preds.reshape((preds.shape[0], -1))
        # Initialize a dummy "unlabeled" mask with the threshold
        flat_preds_with_treshold = torch.full(
            (preds.shape[0] + 1, flat_preds.shape[-1]), threshold
        )
        flat_preds_with_treshold[1 : preds.shape[0] + 1, :] = flat_preds

        # Get the top mask index for each pixel
        semantic_inds = torch.topk(flat_preds_with_treshold, 1, dim=0).indices.reshape(
            (preds.shape[-2], preds.shape[-1])
        )
        return semantic_inds
    
    def clipseg_segmentation(
        self, model, images: list[Image.Image] | torch.Tensor, device: torch.device
    ):
        image_inputs = self.clipseg_processor(images=images, return_tensors="pt").to(device)
        all_inputs = BatchEncoding(data=dict(
            **image_inputs,
            **self.part_text_encoding,
            output_hidden_states=torch.tensor(True),
            output_attentions=torch.tensor(True)
        ), tensor_type='pt')
        outputs = model(**all_inputs)
        return outputs
    
    def inference(self, image):
        image = image[0]
        c, h, w = image.shape
        with torch.no_grad():
            outputs = self.clipseg_segmentation(
                self.clipseg_model,
                [image],
                self.device,
            )
            upscaled_logits = nn.functional.interpolate(
                outputs.logits[:,:-1,:,:],
                size=(h, w),
                mode="bilinear",
            )
            clipseg_preds = torch.sigmoid(upscaled_logits)
        preds = clipseg_preds.squeeze(0)
        results = [{"sem_seg": preds, "outputs": outputs}]
        return results
    
    def forward(self, batch):
        im_paths, tgt_paths, images, targets = batch  # list[tensor]
        if not self.training:
            return self.inference(images)
        images = [im.to(self.device) for im in images]
        tgt_list = [tgt.to(self.device) for tgt in targets]

        outputs = self.clipseg_segmentation(self.clipseg_model, images, self.device) # shape: [b,n,h,w]
        logits = outputs.logits
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
        # losses = {"loss_sem_seg" : loss}
        return loss