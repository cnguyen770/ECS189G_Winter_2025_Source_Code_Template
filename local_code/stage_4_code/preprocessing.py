import re
from collections import Counter

def clean_tokenize(text):
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    return text.split()

class Vocab:
    def __init__(self, token_lists, min_freq=2,
                 specials=['<pad>','<unk>','<bos>','<eos>']):
        counter = Counter(tok for lst in token_lists for tok in lst)
        self.itos = list(specials) + [tok for tok,c in counter.items() if c>=min_freq]
        self.stoi = {tok:i for i,tok in enumerate(self.itos)}

    def encode(self, tokens):
        return [self.stoi.get(tok, self.stoi['<unk>']) for tok in tokens]

    def decode(self, indices):
        return [self.itos[i] for i in indices]
