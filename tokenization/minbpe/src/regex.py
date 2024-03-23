"""
Byte pair encoding tokenizer similar to what GPT-4 uses.
Extensions to base tokenizer
- handles special tokens
- splits the text using a regex
"""

from .base import Tokenizer, get_stats, merge
import regex as re

# Pattern copied from https://github.com/openai/tiktoken/blob/main/tiktoken_ext/openai_public.py
GPT2_SPLIT_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
GPT4_SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""

class RegexTokenizer(Tokenizer):
    def __init__(self, pattern=None):
        """
        - pattern to split the text by, default is gpt-4 pattern
        """
        super().__init__()
        self.pattern = GPT4_SPLIT_PATTERN if pattern is None else pattern
        self.compiled_pattern = re.compile(self.pattern)
        self.special_tokens = {} # dict from str -> int reprsenting special tokens
        self.inverse_special_tokens = {} # dict from int -> str reprsenting special tokens
    
    def register_special_tokens(self, special_tokens):
        # special tokens is a dict from str -> int
        self.special_tokens = special_tokens
        self.inverse_special_tokens = {v:k for k, v in self.special_tokens.items()}
    
    def train(self, text, vocab_size, verbose=False):
        assert vocab_size >= 256
        num_merges = vocab_size - 256
        vocab = {idx: bytes([idx]) for idx in range(256)}

        text_chunks = re.findall(self.compiled_pattern, text)
        ids = [list(chunk.encode('utf-8')) for chunk in text_chunks]
        merges = {}
        
        for i in range(num_merges):
            # maintain global count of occurences of consecutive pairs
            stats = {}
            for chunk_id in ids:
                get_stats(chunk_id, stats)
            
            # find the pair which occurs most consecutively
            max_pair = max(stats, key=stats.get)
            new_id = 256 + i
            ids = [merge(chunk_id, max_pair, new_id) for chunk_id in ids]
            merges[max_pair] = new_id
            vocab[new_id] = vocab[max_pair[0]] + vocab[max_pair[1]]

            # print stats
            if verbose:
                print(f"merge {i+1}/{num_merges}: {max_pair} -> {new_id} ({vocab[new_id]}) had {stats[max_pair]} occurences")
            
        self.merges = merges # used in encode()
        self.vocab = vocab # used in decode()
        # print(f"merges dict size is {len(self.merges)}, num merges is {num_merges}")
    
    def decode(self, ids):
        # return python string given list of integers
        part_bytes = []
        for idx in ids:
            if idx in self.vocab:
                part_bytes.append(self.vocab[idx])
            elif idx in self.inverse_special_tokens:  # understand this part thoroughly
                part_bytes.append(self.inverse_special_tokens[idx].encode("utf-8"))
            else:
                raise ValueError(f"Invalid token for decoding: {idx}")
        
        s = b"".join(part_bytes)
        # print(f"raw bytes is {s}")
        return s.decode('utf-8', errors='replace')
    
    def _encode_chunk(self, text_bytes):
        ids = list(text_bytes)
        while len(ids) >= 2:
            # find the element in stats that has the smallest associated value in merges
            stats = get_stats(ids)
            top_pair = min(stats, key=lambda p: self.merges.get(p, float('inf')))
            if top_pair not in self.merges:
                break
            ids = merge(ids, top_pair, self.merges[top_pair])
        
        return ids
    
    def _encode_ordinary(self, text):
        "Encoding that ignores any special tokens"
        text_chunks = re.findall(self.compiled_pattern, text)
        text_bytes = [chunk.encode('utf-8') for chunk in text_chunks]
        encoded_out = []
        for text_byte in text_bytes:
            encoded_out.extend(self._encode_chunk(text_byte))
        
        return encoded_out
    
    def encode(self, text, allowed_special="none_raise"):
        """
        This function handles special tokens
        allowed_special: can be "all"|"none"|"none_raise"
        tiktoken default behavior is none_raise
        """
        special = None
        if allowed_special == "all":
            special = self.special_tokens
        elif allowed_special == "none":
            special = {}
        elif allowed_special == "none_raise":
            special = {}
            assert all(token not in text for token in self.special_tokens)
        elif isinstance(allowed_special, set):
            special = {k: v for k, v in self.special_tokens.items() if k in allowed_special}
        else:
            raise ValueError(f"allowed_special={allowed_special} not understood")
        
        if not special:
            # revert to encode ordinary
            return self._encode_ordinary(text)
        
        # else need to handle special characters
        # enclosing in parenthesis makes it into a capturing group so that special tokens are includes in the output from split 
        special_pattern = "(" + "|".join(re.escape(k) for k in special) + ")"
        special_chunks = re.split(special_pattern, text)
        ids = []
        for chunk in special_chunks:
            if chunk in special:
                # print(f"special chunk: {chunk}, {special[chunk]}")
                ids.append(special[chunk])
            else:
                ids.extend(self._encode_ordinary(chunk))
        
        return ids
