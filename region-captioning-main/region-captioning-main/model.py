import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence

class AlignmentEncoderCNN(nn.Module):
    """
    Vision Encoder for the Alignment Model.
    Processes a set of 20 image regions (1 full, 19 from YOLO) for each image.
    """
    def __init__(self, embed_size):
        super(AlignmentEncoderCNN, self).__init__()
        vgg = models.vgg19(pretrained=True)
        self.features = vgg.features
        self.avgpool = vgg.avgpool
        
        in_features = vgg.classifier[0].in_features
        self.classifier = nn.Sequential(
            nn.Linear(in_features, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, embed_size),
        )

        # Freeze the feature extraction layers
        for param in self.features.parameters():
            param.requires_grad = False

    def forward(self, image_regions):
        # image_regions shape: (batch_size, num_regions, C, H, W)
        batch_size, num_regions, C, H, W = image_regions.shape
        
        # Reshape to process all regions in one go
        image_regions = image_regions.view(batch_size * num_regions, C, H, W)
        
        features = self.features(image_regions)
        features = self.avgpool(features)
        features = torch.flatten(features, 1)
        embeddings = self.classifier(features)
        
        # Reshape back to (batch_size, num_regions, embed_size)
        embeddings = embeddings.view(batch_size, num_regions, -1)
        return embeddings

class GenerativeEncoderCNN(nn.Module):
    """
    Vision Encoder for the Generative Model.
    Processes a single full image to generate a global feature vector.
    """
    def __init__(self, embed_size):
        super(GenerativeEncoderCNN, self).__init__()
        vgg = models.vgg19(pretrained=True)
        self.features = vgg.features
        self.avgpool = vgg.avgpool
        
        in_features = vgg.classifier[0].in_features
        self.classifier = nn.Sequential(
            nn.Linear(in_features, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, embed_size),
        )

        # Freeze the feature extraction layers
        for param in self.features.parameters():
            param.requires_grad = False

    def forward(self, images):
        # images shape: (batch_size, C, H, W)
        features = self.features(images)
        features = self.avgpool(features)
        features = torch.flatten(features, 1)
        embeddings = self.classifier(features)
        return embeddings

class EncoderBRNN(nn.Module):
    """
    BRNN Encoder (Language Encoder)
    Uses a Bidirectional GRU to create contextual word embeddings.
    """
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers, word_embed_dim=300):
        super(EncoderBRNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, word_embed_dim)
        self.gru = nn.GRU(word_embed_dim, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, embed_size)
        self.relu = nn.ReLU()

    def forward(self, captions, lengths):
        embeddings = self.embedding(captions)
        packed = pack_padded_sequence(embeddings, lengths, batch_first=True, enforce_sorted=False)
        outputs, _ = self.gru(packed)
        unpacked, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)
        features = self.fc(unpacked)
        features = self.relu(features)
        return features

class DecoderMRNN(nn.Module):
    """
    MRNN Decoder (Generative Model)
    Generates a caption from a global image feature vector.
    """
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super(DecoderMRNN, self).__init__()
        self.word_embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)
        self.init_h = nn.Linear(embed_size, hidden_size)   # or feature_dim if encoder output is not embed_size
        self.init_c = nn.Linear(embed_size, hidden_size)

    def forward(self, image_features, captions, lengths):
        batch_size = image_features.size(0)

        # Init hidden & cell states
        num_layers = self.lstm.num_layers
        h0 = self.init_h(image_features).unsqueeze(0).repeat(num_layers, 1, 1)
        c0 = self.init_c(image_features).unsqueeze(0).repeat(num_layers, 1, 1)

        # Embed captions (shift right to exclude <end>)
        word_embeddings = self.word_embedding(captions)   # [B, max_len, E]
        embeddings = word_embeddings[:, :-1, :]
        lengths = [l - 1 for l in lengths]                # <-- important

        # Pack padded sequence
        packed_embeddings = pack_padded_sequence(
            embeddings, lengths, batch_first=True, enforce_sorted=False
        )

        # Decode
        outputs, _ = self.lstm(packed_embeddings, (h0, c0))
        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)

        predictions = self.fc(outputs)
        return predictions



    def sample(self, image_features, vocab, max_len=20):
        """Generate a caption for a single image feature vector."""
        self.eval()
        caption_ids = []
        
        with torch.no_grad():
            # h, c = image_features.unsqueeze(0), torch.zeros_like(image_features.unsqueeze(0))
            h = self.init_h(image_features).unsqueeze(0)
            c = self.init_c(image_features).unsqueeze(0)

            # Start with the <start> token
            start_token_idx = vocab('<start>')
            inputs = self.word_embedding.weight.new_tensor([start_token_idx], dtype=torch.long).unsqueeze(1)
            inputs = self.word_embedding(inputs)

            for _ in range(max_len):
                outputs, (h, c) = self.lstm(inputs, (h, c))
                predictions = self.fc(outputs.squeeze(1))
                predicted_idx = predictions.argmax(1)
                
                caption_ids.append(predicted_idx.item())
                
                if vocab.idx2word[predicted_idx.item()] == '<end>':
                    break
                
                inputs = self.word_embedding(predicted_idx.unsqueeze(0))
                
        return caption_ids