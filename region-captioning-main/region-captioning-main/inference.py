import torch
from torchvision import transforms
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import numpy as np

# ------------------
# Helper: load image
# ------------------
def load_image(image_path):
    return Image.open(image_path).convert("RGB")

# ------------------
# Preprocess region for encoder
# ------------------
def preprocess_region(region):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225))
    ])
    return transform(region).unsqueeze(0)

# ------------------
# Caption one region
# ------------------
def caption_region(region, encoder, decoder, vocab, device):
    region = preprocess_region(region).to(device)
    with torch.no_grad():
        features = encoder(region)
        sampled_ids = decoder.sample(features, vocab)

    caption_words = []
    for word_id in sampled_ids:
        word = vocab.idx2word[word_id]
        if word == "<end>":
            break
        if word != "<start>":
            caption_words.append(word)

    return " ".join(caption_words)

# ------------------
# Main: Generate captions for YOLO regions
# ------------------
def generate_region_captions(image_path, yolo_model, encoder, decoder, vocab, device, conf_thresh=0.3):
    """
    Detect regions with YOLO, then generate captions for each region.
    Returns image with drawn boxes + captions and list of region captions.
    """
    image = load_image(image_path)
    results = yolo_model(image)

    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    region_captions = []

    for det in results.xyxy[0]:  # xyxy: (x1, y1, x2, y2, conf, cls)
        x1, y1, x2, y2, conf, cls = det.tolist()
        if conf < conf_thresh:
            continue

        # Crop region
        region = image.crop((x1, y1, x2, y2))

        # Caption region
        caption = caption_region(region, encoder, decoder, vocab, device)
        region_captions.append(((x1, y1, x2, y2), caption))

        # Draw on original image
        draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
        draw.text((x1, y1 - 10), caption, fill="red", font=font)

    return image, region_captions
