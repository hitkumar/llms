from typing import Optional, List
import torch
import time
from pathlib import Path
import json
from sentencepiece import SentencePieceProcessor
from tqdm import tqdm

from model import ModelArgs, Transformer

class LLaMA:
    def __init__(self, model: Transformer, tokenizer: SentencePieceProcessor, model_args: ModelArgs):
        self.model = model
        self.tokenizer = tokenizer
        self.args = model_args
    
    def _sample_top_p(self, probs, p):
        # (B, vocab_size)
        probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
        # (B, vocab_size)
        probs_sum = torch.cumsum(probs_sort, dim=-1)
        # (B, vocab_size)
        mask = probs_sum - probs_sort > p
        probs_sort[mask] = 0.0

        probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
        # (B, 1)
        next_token = torch.multinomial(probs_sort, num_samples=1)
        next_token = torch.gather(probs_idx, -1, next_token)
        return next_token
    
    @staticmethod
    def build(checkpoints_dir: str, tokenizer_path: str, load_model: bool, max_seq_len: int, max_batch_size: int, device: str):
        prev_time = time.time()
        if load_model:
            checkpoints = sorted(Path(checkpoints_dir).glob("*.pth"))
            assert len(checkpoints) > 0, f"no checkpoints found in {checkpoints_dir}"
            chpt_path = checkpoints[0]
            print(f'Loading checkpoint "{chpt_path}"')
            checkpoint = torch.load(chpt_path, map_location="cpu")
            print(f'Loaded checkpoint in {time.time() - prev_time:.2f}s')
            # reset time now
            prev_time = time.time()
        
        with open(Path(checkpoints_dir)/"params.json", "r") as f:
            params = json.loads(f.read())
            
        model_args: ModelArgs = ModelArgs(
            max_seq_len=max_seq_len,
            max_batch_size=max_batch_size,
            device=device,
            **params
        )

        # Load tokenizer
        tokenizer = SentencePieceProcessor()
        tokenizer.load(tokenizer_path)
        model_args.vocab_size = tokenizer.vocab_size()

        if device == "cuda":
            torch.set_default_tensor_type(torch.cuda.HalfTensor)
        else:
            torch.set_default_tensor_type(torch.BFloat16Tensor)
        
        model = Transformer(model_args).to(device)

        if load_model:
            # we don't match rope.freqs
            del checkpoint['rope.freqs']
            model.load_state_dict(checkpoint, strict=True)
            print(f"Loaded state dict in {time.time() - prev_time:.2f}s")
        
        return LLaMA(
            model, tokenizer, model_args
        )

    def text_completion(self, prompts: List[str], temperature: float = 0.6, top_p: float = 0.9, max_gen_len: Optional[int] = None):
        if max_gen_len is None:
            max_gen_len = self.args.max_seq_len - 1
        
        prompt_tokens = [self.tokenizer.encode(prompt, out_type=int, add_bos=True, add_eos=False) for prompt in prompts]
        batch_size = len(prompt_tokens)
        assert batch_size <= self.args.max_batch_size, "Batch size too large"

        max_prompt_len = max(len(prompt_token) for prompt_token in prompt_tokens)
        assert max_prompt_len <= self.args.max_seq_len, "Prompt is too large"

        total_len = min(self.args.max_seq_len, max_prompt_len + max_gen_len)
        pad_id = self.tokenizer.pad_id()

        # this the array we will fill gradually with generated tokens
        tokens = torch.full((batch_size, total_len), pad_id, dtype=torch.long, device=device)

        # fill array with prompt tokens so far
        for prompt_id, current_prompt in enumerate(prompt_tokens):
            tokens[prompt_id, :len(current_prompt)] = torch.tensor(current_prompt, dtype=torch.long, device=device)
        
        eos_reached = torch.tensor([False] * batch_size, device=device)
        prompt_token_mask = tokens != pad_id

        curr_iter = tqdm(range(1, total_len), desc='Generating tokens')
        for curr_pos in curr_iter:
            with torch.no_grad():
                logits = self.model(tokens[:, curr_pos-1:curr_pos], curr_pos)
                print(logits.shape)
            
            # Top-p decoding or greedy decoding
            if temperature > 0:
                probs = torch.softmax(logits[:, -1] / temperature, dim=-1)
                next_token = self._sample_top_p(probs, top_p)
            else:
                next_token = torch.argmax(logits[:, -1], dim=-1)
            
            next_token = next_token.reshape(-1)
            print(f"next token shape is {next_token.shape}")

            next_token = torch.where(prompt_token_mask[:, curr_pos], tokens[:, curr_pos], next_token)
            tokens[:, curr_pos] = next_token

            eos_reached |= (~prompt_token_mask[:, curr_pos]) & (next_token == self.tokenizer.eos_id)
            if all(eos_reached):
                break
        
        # generate the decoded list of tokens
        out_tokens = []
        out_text = []
        for i, token_list in enumerate(tokens.tolist()):
            # Only consider until eos token
            if self.tokenizer.eos_id in token_list:
                eos_idx = token_list.index(self.tokenizer.eos_id)
                print(eos_idx)
                token_list = token_list[:eos_idx]
    
            out_tokens.append(token_list)
            out_text.append(self.tokenizer.decode(token_list))
        
        return (out_tokens, out_text)

        
if __name__ == '__main__':
    torch.manual_seed(0)

    allow_cuda = False
    device = 'cuda' if torch.cuda.is_available() and allow_cuda else 'cpu'

    model = LLaMA.build(
        checkpoints_dir='llama-2-7b/',
        tokenizer_path='tokenizer.model',
        load_model=True,
        max_seq_len=1024,
        max_batch_size=12,
        device=device
    )

    prompts = [
        "How are you doing?"
    ]
    out_tokens, out_texts = model.text_completion(prompts, max_gen_len=12)

    assert len(out_texts) == len(prompts)
    print(out_texts)