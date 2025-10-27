import os
import torch
import pandas as pd
import nltk
from torch.utils.data import Dataset
from PIL import Image

class FlickrDataset(Dataset):
    """
    Flickr Dataset for Image Captioning.
    Reads data based on the specified file structure:
    - root_dir/
      - Images/ (all.jpg images)
      - captions.txt (CSV with 'image' and 'caption' headers)
    """
    def __init__(self, root_dir, captions_file, vocab, transform=None, stage='alignment'):
        self.root_dir = root_dir
        self.transform = transform
        self.vocab = vocab
        self.stage = stage # 'alignment' or 'generative'

        # Load captions from captions.txt
        self.df = pd.read_csv(captions_file)
        
        # Group captions by image
        self.image_captions = self.df.groupby('image')['caption'].apply(list).to_dict()
        self.image_names = list(self.image_captions.keys())

        # Create a flat list of all (image, caption) pairs
        self.samples = []
        for img_name, captions_list in self.image_captions.items():
            for caption in captions_list:
                self.samples.append((img_name, caption))
        
        # Load YOLO model for the alignment stage
        if self.stage == 'alignment':
            self.yolo_model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
            self.yolo_model.eval() # Set to evaluation mode

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_name, caption = self.samples[idx]
        img_path = os.path.join(self.root_dir, 'Images', img_name)
        
        image = Image.open(img_path).convert('RGB')

        # --- Stage 1: Alignment Model Data ---
        if self.stage == 'alignment':
            # Use YOLO to get bounding boxes
            with torch.no_grad():
                results = self.yolo_model(image)
            
            bboxes = results.xyxy[0].cpu().numpy()
            
            # Get top 19 boxes by confidence, plus the whole image
            top_bboxes = bboxes[bboxes[:, 4].argsort()[-19:]]

            regions = []
            # Add the full image first
            regions.append(image)
            
            # Add cropped regions from YOLO detections
            for bbox in top_bboxes:
                x1, y1, x2, y2, conf, _ = bbox
                region = image.crop((x1, y1, x2, y2))
                regions.append(region)
            
            # Pad with full image if less than 20 regions found
            while len(regions) < 20:
                regions.append(image)

            # Apply transformations to all 20 regions
            if self.transform:
                image_regions = torch.stack([self.transform(r) for r in regions])
            
            output_image = image_regions

        # --- Stage 2: Generative Model Data ---
        else: # self.stage == 'generative'
            if self.transform:
                output_image = self.transform(image)

        # Convert caption (string) to word ids.
        tokens = nltk.tokenize.word_tokenize(str(caption).lower())
        caption_vec = []
        caption_vec.append(self.vocab('<start>'))
        caption_vec.extend([self.vocab(token) for token in tokens])
        caption_vec.append(self.vocab('<end>'))
        target = torch.Tensor(caption_vec)
        
        return output_image, target

def collate_fn(data):
    """Creates mini-batch tensors from the list of tuples (image/regions, caption)."""
    data.sort(key=lambda x: len(x[1]), reverse=True)
    images, captions = zip(*data)

    # Merge images (from tuple of tensors to a single tensor).
    images = torch.stack(images, 0)

    # Merge captions (from tuple of 1D tensor to 2D tensor).
    lengths = [len(cap) for cap in captions]
    targets = torch.zeros(len(captions), max(lengths)).long()
    for i, cap in enumerate(captions):
        end = lengths[i]
        targets[i, :end] = cap[:end]
        
    return images, targets, lengths