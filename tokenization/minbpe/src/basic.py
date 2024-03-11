from .base import Tokenizer, get_stats, merge

class BasicTokenizer(Tokenizer):
    def __init__(self):
        super().__init__()
    
    def train(self, text, vocab_size, verbose=False):
        assert vocab_size >= 256
        num_merges = vocab_size - 256
        ids = list(text.encode('utf-8'))

        merges = {}
        vocab = {idx: bytes([idx]) for idx in range(256)}
        idx = 256
        for i in range(num_merges):
            stats = get_stats(ids)
            top_pair = max(stats, key=stats.get)
            ids = merge(ids, top_pair, idx)
            merges[top_pair] = idx
            vocab[idx] = vocab[top_pair[0]] + vocab[top_pair[1]]
            if verbose:
                print(f"merge {i+1}/{merges}: {top_pair} -> {idx} {vocab[idx]} has {stats[top_pair]} occurences")
            idx +=1
        
        self.merges = merges
        self.vocab = vocab
    
    def decode(self, ids):
        """Converts ids to a string"""
        text_bytes = b"".join(self.vocab[idx] for idx in ids)
        text = text_bytes.decode("utf-8", errors="replace")
        return text
    
    def encode(self, text):
        """Retums ids from text"""
        ids = list(text.encode('utf-8'))
        while len(ids) >= 2:
            # find the element in stats that has the smallest associated value in merges
            stats = get_stats(ids)
            top_pair = min(stats, key=lambda p: self.merges.get(p, float('inf')))
            if top_pair not in self.merges:
                break
            ids = merge(ids, top_pair, self.merges[top_pair])
        
        return ids