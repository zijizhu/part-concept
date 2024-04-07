from PIL import Image
import torch
from torch import nn
import torch.nn.functional as F
from clipseg.processing_clipseg import CLIPSegProcessor
from clipseg.modeling_clipseg import CLIPSegForImageSegmentation


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


class CLIPSeg(nn.Module):
    def __init__(self):
        super().__init__()
        self.clipseg_processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
        train_class_texts = BASE_PARTS_NAMES
        self.train_class_texts = [c.replace('\'s', '') for c in train_class_texts]
        self.train_obj_classes = OBJ_BASE_CLASS_NAMES

        self.test_class_texts = PARTS_NAMES
        self.test_obj_classes = OBJ_CLASS_NAMES
        
        self.segmentation_background_threshold = 0.0
        self.clipseg_model = CLIPSegForImageSegmentation.from_pretrained(
            "CIDAS/clipseg-rd64-refined"
        )
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.ignore_label = 255
        
        for name, params in self.clipseg_model.named_parameters():
            if 'clip.text_model.embeddings' in name or 'film' in name or 'visual_adapter' in name or 'decoder' in name: # VA+L+F+D
                params.requires_grad = True
            else:
                params.requires_grad = False
        self.train_text_encoding = self.clipseg_processor.tokenizer(self.train_class_texts, return_tensors="pt", padding="max_length")
        
    @classmethod
    def from_config(cls, cfg):
        ret = {}
        ret["train_dataset"] = cfg.DATASETS.TRAIN[0]
        ret['test_dataset'] = cfg.DATASETS.TEST[0]
        return ret
    
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
        self, model, images, test_text, device
    ):
        logits = []
        input = self.clipseg_processor(images=images, return_tensors="pt").to(device)
        if self.training:
            text = self.train_text_encoding
        else:
            text=  test_text
        input.update(text)
        input.update({'output_hidden_states': torch.tensor(True), 'output_attentions': torch.tensor(True)})
        input = input.to(device)
        outputs = model(**input)
        # logits = outputs.logits
        # return logits
        return outputs
    
    def inference(self, batched_inputs):
        image = Image.open(batched_inputs[0]["file_name"])
        image = image.convert("RGB")
        with torch.no_grad():
            outputs = self.clipseg_segmentation(
                self.clipseg_model,
                [image],
                self.clipseg_processor.tokenizer([part.replace('\'s', '') for part in self.test_class_texts], return_tensors="pt", padding="max_length"),
                self.device,
            )
            logits = outputs.logits
            upscaled_logits = nn.functional.interpolate(
                            logits[:,:-1,:,:],
                            size=(image.size[1], image.size[0]),
                            mode="bilinear",
                            )
            clipseg_preds = torch.sigmoid(upscaled_logits)
        gt_objs = [self.test_obj_classes[i] for i in torch.unique(batched_inputs[0]["sem_seg"]) if i != self.ignore_label]
        part_category_names = []
        part_inds = []
        for obj in gt_objs:
            for i,part in enumerate(self.test_class_texts):
                if part.split('\'s')[0] == obj:
                    part_category_names.append(part.replace('\'s', ''))
                    part_inds.append(i)
        no_part_ids = [i for i in range(len(self.test_class_texts)) if i not in part_inds]  
        preds = clipseg_preds.squeeze(0)
        preds[no_part_ids] = 0.0
        results = [{"sem_seg": preds, "outputs": outputs}]
        return results
    
    def forward(self, batched_inputs):
        if not self.training:
            return self.inference(batched_inputs)
        # images = [Image.open(x["file_name"]).convert("RGB") for x in batched_inputs]
        images = [x["image"].to(self.device) for x in batched_inputs]
        gts = [x["obj_part_sem_seg"].to(self.device) for x in batched_inputs]
        outputs = self.clipseg_segmentation(self.clipseg_model, images, None, self.device) #[b,n,h,w]
        outputs = outputs.logits
        targets = torch.stack([nn.functional.interpolate(
                gt.unsqueeze(0).unsqueeze(0).float(),
                size=(outputs.shape[-2], outputs.shape[-1]),
                mode="nearest") for gt in gts]).long().squeeze(1).squeeze(1) #[b,h,w]

        num_classes = outputs.shape[1]
        mask = targets != self.ignore_label #[b,h,w]
        outputs = outputs.permute(0,2,3,1) #[b,h,w,n]
        _targets = torch.zeros(outputs.shape, device=self.device)
        class_weight = torch.ones(num_classes).cuda()
        _targets[:,:,:,-1] = 1
        class_weight[-1] = 0.05
        _onehot = F.one_hot(targets[mask], num_classes=num_classes).float()
        _targets[mask] = _onehot

        loss = F.binary_cross_entropy_with_logits(outputs, _targets, weight=class_weight)
        losses = {"loss_sem_seg" : loss}
        return losses