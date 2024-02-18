from pathlib import Path
from dataclasses import dataclass

@dataclass
class ModelConfig:
    batch_size: int = 8
    num_epochs: int = 2
    lr: float= 1e-4
    seq_len: int =  350
    d_model: int = 512
    datasource: str = 'opus_books'
    lang_src: str = 'en'
    lang_tgt: str = 'it'
    model_folder: str = 'weights'
    model_basename: str = 'tmodel_'
    preload: str = 'latest'
    tokenizer_file: str = "tokeninzer_{0}.json"
    experiment_name: str = 'runs/tmodel'

def get_weights_file_path(config: ModelConfig, epoch: str):
    model_folder = f"{config.datasource}_{config.model_folder}"
    model_filename = f"{config.model_basename}{epoch}.pt"
    return str(Path('.')/model_folder/model_filename)

def latest_weights_file_path(config):
    model_folder = f"{config.datasource}_{config.model_folder}"
    # print(model_folder)
    model_filename = f"{config.model_basename}*"
    weight_files = list(Path(model_folder).glob(model_filename))

    if len(weight_files) == 0:
        return None
    
    weight_files.sort()
    weights_file = str(weight_files[-1])
    print(f"loading {weights_file}")
    return weights_file

def get_config():
    return ModelConfig()