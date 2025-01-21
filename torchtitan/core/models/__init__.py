from core.models.llama import llama3_configs, Transformer

models_config = {
    "llama3": llama3_configs,
}

model_name_to_cls = {"llama3": Transformer}

model_name_to_tokenizer = {
    "llama3": "tiktoken",
}
