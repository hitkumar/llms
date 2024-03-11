import unicodedata
from pathlib import Path

def get_stats(ids):
    counts = {}
    for pair in zip(ids, ids[1:]):
        counts[pair] = counts.get(pair, 0) + 1
    return counts

def merge(ids, pair, idx):
    new_ids = []
    i = 0
    while i < len(ids):
        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i+1] == pair[1]:
            new_ids.append(idx)
            i += 2
        else:
            new_ids.append(ids[i])
            i += 1
    
    return new_ids

def replace_control_characters(s: str) -> str:
    chars = []
    for ch in s:
        if unicodedata.category(ch)[0] != 'C':
            chars.append(ch)
        else:
            chars.append(f"\\u{ord(ch):04x}") # escape
    
    return "".join(chars)

def render_token(t: bytes) -> str:
    s = t.decode('utf-8', errors='replace')
    return replace_control_characters(s)


# the base tokenizer class

class Tokenizer:
    """Base class for tokenizer"""

    def __init__(self):
        # default vocab size is 256 (same as ascii chars), no merges, no patterns
        self.merges = {} # (int, int) -> int
        self.pattern = "" # str
        self.special_tokens = {} # str -> int eg. {'<|endoftext|>': 1}
        self.vocab = self._build_vocab() # int -> bytes
    
    def _build_vocab(self):
        vocab = {idx: bytes(idx) for idx in range(256)}
        # the fact that iteration order is same as order in which items are inserted is key here, otherwise we don't have vocab entries for previous merges
        for (p0, p1), idx in self.merges.items():
            vocab[idx] = vocab[p0] + vocab[p1]
        for special, idx in self.special_tokens.items():
            vocab[idx] = special.encode('utf-8')
        
        return vocab
    
    def train(self, text, vocab_size, verbose=False):
        raise NotImplementedError
    
    def encode(self, text):
        raise NotImplementedError
    
    def decode(self, ids):
        raise NotImplementedError
    
    def save(self, file_prefix):
        """
        Saves two files: file_prefix.vocab and file_prefix.model
        Similar to sentencepiece
        - model file is used for model loading, vocab is just for human viz.
        """
        file = Path(file_prefix)
        model_file = file.with_suffix('.model')
        with open(model_file, 'w') as f:
            # write version, pattern and merges
            f.write('minbpe v1\n')
            f.write(f"{self.pattern}\n")
            # special tokens
            f.write(f"{len(self.special_tokens)}\n")
            for special, idx in self.special_tokens.items():
                f.write(f"{special} {idx}\n")
            
            # merges dict
            for idx1, idx2 in self.merges: # write only the ids of the merge
                f.write(f"{idx1} {idx2}\n")
        
        # write the vocab, for human viz
        # vocab file is different than actual vocab, file is lossy but self.vocab is good.
        vocab_file = file.with_suffix('.vocab')
        inverted_merges = {idx: pair for pair, idx in self.merges.items()}
        with open(vocab_file, 'w', encoding='utf-8') as f:
            for idx, token in self.vocab.items():
                # replaces some partial utf-8 seq into ? token, so this can't be decoded due to error = 'replace'
                s = render_token(token)
                if idx in inverted_merges:
                    idx0, idx1 = inverted_merges[idx]
                    s0 = render_token(self.vocab[idx0])
                    s1 = render_token(self.vocab[idx1])
                    f.write(f"[{s0}][{s1}] -> [{s}] {idx}\n") # we should be able to change this
                else:
                    # print the bytes and special characters, double check the special characters part.
                    f.write(f"[{s}] {idx}")
    
    def load(self, model_file):
        """Invert the functionality in save, but only for model file"""
        assert model_file.endswith(".model")
        # read the model file
        merges = {}
        special_tokens = {}
        idx = 256

        with open(model_file, 'r', encoding='utf-8') as f: # this is decoding, but understand this part more.
            version = f.readline().strip()
            assert version == "minbpe v1"
            self.pattern = f.readline().strip()
            num_special = int(f.readline().strip())
            for _ in range(num_special):
                special, special_idx = f.readline().strip().split()
                special_tokens[special] = int(special_idx)

            # read the merges
            for line in f:
                idx1, idx2 = map(int, line.split())
                merges[(idx1, idx2)] = idx
                idx += 1
        
        self.merges = merges
        self.special_tokens = special_tokens
        self.vocab = self._build_vocab()
