import nltk
import pickle
from collections import Counter

class Vocabulary:
    """
    Simple Vocabulary wrapper
    """
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0
    
    def add_word(self, word):
        if word not in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1
    
    def __call__(self, word):
        if word not in self.word2idx:
            return self.word2idx['<unk>']
        return self.word2idx[word]

    def __len__(self):
        return len(self.word2idx)

def build_vocab(tokens_file, threshold = 5):
    """
    Build a vocabulary from a file of captions
    """
    counter = Counter()
    with open(tokens_file, 'r') as f:
        lines = f.readlines()
    
    for line in lines:
        parts = line.strip().split()
        caption = ' '.join(parts[1:])
        tokens = nltk.tokenize.word_tokenize(caption.lower())
        counter.update(tokens)
    
    # Filter words with frequency < threshold
    words = [word for word, cnt in counter.items() if cnt >= threshold]

    # create the vocab wrapper and add othern special tokens
    vocab = Vocabulary()
    vocab.add_word('<pad>')
    vocab.add_word('<start>')
    vocab.add_word('<end>')
    vocab.add_word('<unk>')

    # add words from captions to vocab
    for word in words:
        vocab.add_word(word)
    
    return vocab

if __name__ == '__main__':
    # Build and save vocabulary
    vocab = build_vocab(tokens_file='data/tokens.txt')
    with open('data/vocab.pkl', 'wb') as f:
        pickle.dump(vocab, f)
    print(f"Total vocabulary size : {len(vocab)}")
    print(f"Saved the vocabulary to data/vocab.pkl")