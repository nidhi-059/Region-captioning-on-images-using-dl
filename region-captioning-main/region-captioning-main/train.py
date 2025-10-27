import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
from torch.nn.utils.rnn import pack_padded_sequence

def train_alignment_epoch(image_encoder, text_encoder, data_loader, optimizer, device):
    """Train the alignment model for one epoch."""
    image_encoder.train()
    text_encoder.train()
    
    total_loss = 0.0
    margin = 1.0

    for image_regions, captions, lengths in tqdm(data_loader, desc="Training Alignment"):
        image_regions = image_regions.to(device)
        captions = captions.to(device)

        optimizer.zero_grad()

        # Get embeddings
        img_embeds = image_encoder(image_regions) # (B, 20, embed_size)
        text_embeds = text_encoder(captions, lengths) # (B, seq_len, embed_size)

        # Compute scores based on Equation 8 from the paper
        scores = torch.zeros(image_regions.size(0), image_regions.size(0), device=device)
        for i in range(image_regions.size(0)): # For each image in batch
            for j in range(image_regions.size(0)): # For each caption in batch
                img_i_regions = img_embeds[i] # (20, embed_size)
                text_j_words = text_embeds[j][:lengths[j]] # (len_j, embed_size)
                
                # Similarity matrix between regions of image i and words of caption j
                sim_matrix = torch.matmul(img_i_regions, text_j_words.t()) # (20, len_j)
                
                # For each word, find the best matching region (max over regions)
                max_sims, _ = torch.max(sim_matrix, dim=0) # (len_j)
                
                # Total score is the sum of these max similarities
                scores[i, j] = torch.sum(max_sims)

        # Max-margin loss
        diagonal = scores.diag().view(image_regions.size(0), 1)
        d1 = diagonal.expand_as(scores)
        d2 = diagonal.t().expand_as(scores)

        cost_s = (margin + scores - d1).clamp(min=0) # Sentences ranking
        cost_im = (margin + scores - d2).clamp(min=0) # Images ranking

        mask = torch.eye(scores.size(0), device=device) > 0.5
        cost_s = cost_s.masked_fill_(mask, 0)
        cost_im = cost_im.masked_fill_(mask, 0)

        loss = cost_s.sum() + cost_im.sum()
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(image_encoder.parameters(), 5.0)
        torch.nn.utils.clip_grad_norm_(text_encoder.parameters(), 5.0)
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(data_loader)


def train_generative_epoch(encoder, decoder, data_loader, optimizer, criterion, device):
    """Train the generative model for one epoch."""
    encoder.eval() # Encoder is frozen
    decoder.train()

    total_loss = 0.0
    
    for images, captions, lengths in tqdm(data_loader, desc="Training Generative"):
        images = images.to(device)
        captions = captions.to(device)
        
        optimizer.zero_grad()
        
        with torch.no_grad():
            features = encoder(images)
            
        outputs = decoder(features, captions, lengths)
        
              # Align targets with outputs
        targets = pack_padded_sequence(captions[:, 1:], [l - 1 for l in lengths],
                                       batch_first=True, enforce_sorted=False).data
        outputs = pack_padded_sequence(outputs, [l - 1 for l in lengths],
                                       batch_first=True, enforce_sorted=False).data

        loss = criterion(outputs, targets)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()

    return total_loss / len(data_loader)