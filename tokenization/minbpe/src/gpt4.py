"""
Implements gpt4 tokenizer in minbpe framework. This is a pretrained tokenizer which we load from tiktoken and build a ```Tokenizer``` from.
"""
import tiktoken
from .regex import RegexTokenizer

GPT4_SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""

GPT4_SPECIAL_TOKENS = {
    '<|endoftext|>': 100257,
    '<|fim_prefix|>': 100258,
    '<|fim_middle|>': 100259,
    '<|fim_suffix|>': 100260,
    '<|endofprompt|>': 100276
}

def bpe(mergeable_ranks, token, max_rank):
    parts = [bytes([b]) for b in token]
    while True:
        min_idx = None
        min_rank = None
        for i, pair in enumerate(zip(parts[:-1], parts[1:])):
            rank = mergeable_ranks.get(pair[0] + pair[1])
            # print(f"rank is {rank}, pair is {pair}")
            if rank is not None and (min_rank is None or rank < min_rank):
                min_idx = i
                min_rank = rank
        
        if min_rank is None or (max_rank is not None and min_rank >= max_rank):
            break

        assert min_idx is not None
        parts = parts[:min_idx] + [parts[min_idx] + parts[min_idx + 1]] + parts[min_idx + 2:]
        # print(f"min_rank is {min_rank}, min_idx is {min_idx}, parts is {parts}")
    return parts

def recover_merges(mergeable_ranks):
    """
    mergeable_ranks is a map from byte seq -> int.
    in this func, we are recovering the original pairings.
    We do BPE training on all the tokens to recover this.
    """
    merges = {}
    for token, rank in mergeable_ranks.items():
        if len(token) == 1:
            continue
        pair = tuple(bpe(mergeable_ranks, token, rank))
        assert len(pair) == 2
        # recover the ranks
        id1 = mergeable_ranks[pair[0]]
        id2 = mergeable_ranks[pair[1]]
        merges[(id1, id2)] = rank
    
    return merges

class GPT4Tokenizer(RegexTokenizer):
    def __init__(self):
        super().__init__(pattern=GPT4_SPLIT_PATTERN)
        # get the official tokenizer
        enc = tiktoken.get_encoding('cl100k_base')
        mergeable_ranks = enc._mergeable_ranks
        # recover merges
        self.merges = recover_merges(mergeable_ranks)
        # print(self.merges)
        # build vocab object
        self.vocab = {id: bytes([id]) for id in range(256)}
        for (p0, p1), rank in self.merges.items():
            self.vocab[rank] = self.vocab[p0] + self.vocab[p1]
        
        # continue with the tricky part
        # some individual token bytes are permuted which we need to deal with here.
        self.bytes_shuffle = {i: mergeable_ranks[bytes([i])] for i in range(256)}
        # bytes_shuffle
        self.inverse_bytes_shuffle = {v: k for k, v in self.bytes_shuffle.items()}
        # self.register_special_tokens(GPT4_SPECIAL_TOKENS)
    
    def _encode_chunk(self, text_bytes):
        # we need to shuffle as merges uses the shuffled ids
        text_bytes = bytes(self.bytes_shuffle[b] for b in text_bytes)
        return super()._encode_chunk(text_bytes)
    
    def decode(self, ids):
        # part_bytes = []
        # for idx in ids:
        #     if idx in self.vocab:
        #         part_bytes.append(self.vocab[idx])
        #     elif idx in self.inverse_special_tokens:  # understand this part thoroughly
        #         print(f"In special tokens decode: {idx}")
        #         part_bytes.append(self.inverse_special_tokens[idx].encode("utf-8"))
        #     else:
        #         raise ValueError(f"Invalid token for decoding: {idx}")
        
        # text_bytes = b"".join(part_bytes)
        # print(list(text_bytes))
        # text_bytes = bytes(self.inverse_bytes_shuffle[i] for i in text_bytes)

        # # print(f"raw bytes is {s}")
        # return text_bytes.decode('utf-8', errors='replace')
    
        # Karpathy's implementation (doesn't work for special tokens)
    
        text_bytes = b"".join(self.vocab[idx] for idx in ids)
        text_bytes = bytes(self.inverse_bytes_shuffle[i] for i in text_bytes)
        return text_bytes.decode('utf-8', errors='replace')

        # test if we can inverse shuffle first and then decode, seems simpler as we convert to bytes once then.
        """ we can't do inverse shuffle first and then decode as inverse shuffle only contains ids for individual tokens, 
        so we first need to find ids for individual tokens from vocab, then inverse byte shuffle and then decode"""

        # return super().decode(list(text_bytes))
        # text_ids_diff = [self.inverse_bytes_shuffle[id] for id in ids]
        # text_bytes_diff = b"".join(self.vocab[idx] for idx in text_ids_diff)
        # print(f"correct text bytes is {text_bytes}, diff bytes is {text_bytes_diff}")
        # return text_bytes.decode('utf-8', errors='replace')
    
    # pretrained tokenizer, can't be implemented.
    def train(self, text, vocab_size, verbose=False):
        raise NotImplementedError
    
    def save(self, file_prefix):
        raise NotImplementedError('GPT4Tokenizer can\'t be saved')
    
    def load(self, model_file):
        raise NotImplementedError()
    
    def save_vocab(self, vocab_file):
        from .base import render_token
        # what are we doing here, not fully clear.
        vocab = {idx: bytes([self.inverse_bytes_shuffle[idx]]) for idx in range(256)}
        # print(vocab)
        for (p0, p1), idx in self.merges.items():
            vocab[idx] = vocab[p0] + vocab[p1]
        
        inverted_merges = {idx: pair for pair, idx in self.merges.items()}
        with open(vocab_file, 'w', encoding='utf-8') as f:
            for idx, token in vocab.items():
                # replaces some partial utf-8 seq into ? token, so this can't be decoded due to error = 'replace'
                s = render_token(token)
                if idx in inverted_merges:
                    idx0, idx1 = inverted_merges[idx]
                    s0 = render_token(vocab[idx0])
                    s1 = render_token(vocab[idx1])
                    f.write(f"[{s0}][{s1}] -> [{s}] {idx}\n") # we should be able to change this
                else:
                    # print the bytes and special characters, double check the special characters part.
                    f.write(f"[{s}] {idx}\n")