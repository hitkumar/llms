import fire
import torch
from modeling_gemma import KVCache, PaliGemmaConfig, PaliGemmaForConditionalGeneration
from paligemma_utils import load_hf_model
from PIL import Image
from processing_paligemma import PaliGemmaProcessor


def move_inputs_to_device(model_inputs: dict, device: str):
    model_inputs = {k: v.to(device) for k, v in model_inputs.items()}
    return model_inputs


def get_model_inputs(
    processor: PaliGemmaProcessor, prompt: str, image_file_path: str, device: str
):
    image = Image.open(image_file_path)
    model_inputs = processor([prompt], [image])
    model_inputs = move_inputs_to_device(model_inputs, device)
    return model_inputs


def test_inference(
    model: PaliGemmaForConditionalGeneration,
    processor: PaliGemmaProcessor,
    prompt: str,
    image_file_path: str,
    device: str,
    max_tokens_to_generate: int,
    temperature: float,
    top_p: float,
    do_sample: bool,
):
    model_inputs = get_model_inputs(processor, prompt, image_file_path, device)
    input_ids = model_inputs["input_ids"]
    pixel_values = model_inputs["pixel_values"]
    attention_mask = model_inputs["attention_mask"]
    print(input_ids.dtype)

    kv_cache = KVCache()

    stop_token = processor.tokenizer.eos_token_id
    generated_tokens = []

    for _ in range(max_tokens_to_generate):
        # [B, seq_len, vocab_size]
        outputs = model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            attention_mask=attention_mask,
            kv_cache=kv_cache,
        )
        kv_cache = outputs["kv_cache"]
        # [B, vocab_size]
        next_token_logits = outputs["logits"][:, -1, :]
        if do_sample:
            # TODO: add support for top_p
            print("test")
        else:
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

        assert next_token.shape == (1, 1)
        next_token = next_token.squeeze(0)
        generated_tokens.append(next_token)

        if next_token == stop_token:
            break

        # shape is (1, 1) we can discard the previous input because of kv cache
        input_ids = next_token.unsqueeze(-1)
        # [B, seq_len + 1]
        attention_mask = torch.cat(
            [attention_mask, torch.ones((1, 1), device=input_ids.device)], dim=-1
        )

    generated_tokens = torch.cat(generated_tokens, dim=-1)
    # TODO: find out why skipping special tokens
    decoded = processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)

    print(prompt + decoded)
