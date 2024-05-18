from dataclasses import dataclass
import os
import requests
import json
import numpy as np
import tensorflow as tf
from tqdm import tqdm


def download_file(file_path, destination):
    response = requests.get(file_path, stream=True)
    file_size = int(response.headers.get("content-length", 0))
    if os.path.exists(destination):
        file_size_local = os.path.getsize(destination)
        if file_size == file_size_local:
            print(f"File exists: {destination}")
            return

    block_size = 1024
    progress_bar_description = file_path.split("/")[-1]
    with tqdm(total=file_size, unit='iB', unit_scale=True, desc=progress_bar_description) as progress_bar:
        with open(destination, "wb") as file:
            for chunk in response.iter_content(block_size):
                progress_bar.update(len(chunk))
                file.write(chunk)

def load_gpt2_params_from_tf_ckpt(ckpt_path, settings):
    # Initialize parameters dictionary with empty blocks for each layer
    params = {"blocks": [{} for _ in range(settings["n_layer"])]}

    # Iterate over each variable in the checkpoint
    for name, _ in tf.train.list_variables(ckpt_path):
        # Load the variable and remove singleton dimensions
        variable_array = np.squeeze(tf.train.load_variable(ckpt_path, name))

        # Process the variable name to extract relevant parts
        variable_name_parts = name.split("/")[1:]  # Skip the 'model/' prefix

        # Identify the target dictionary for the variable
        target_dict = params
        if variable_name_parts[0].startswith("h"):
            layer_number = int(variable_name_parts[0][1:])
            target_dict = params["blocks"][layer_number]

        # Recursively access or create nested dictionaries
        for key in variable_name_parts[1:-1]:
            target_dict = target_dict.setdefault(key, {})

        # Assign the variable array to the last key
        last_key = variable_name_parts[-1]
        target_dict[last_key] = variable_array

    return params

def download_and_load_gpt2(model_size, models_dir):
    # model sizes available in https://github.com/openai/gpt-2/blob/master/DEVELOPERS.md
    allowed_sizes = ('124M', '355M', '774M', '1558M')
    if model_size not in allowed_sizes:
        raise ValueError(f"Model size {model_size} not in {allowed_sizes}")
    
    # define paths
    model_dir = os.path.join(models_dir, model_size)
    base_url = "https://openaipublic.blob.core.windows.net/gpt-2/models"
    filenames = [
        'checkpoint','encoder.json','hparams.json',
        'model.ckpt.data-00000-of-00001', 'model.ckpt.index',
        'model.ckpt.meta', 'vocab.bpe'
    ]

    # download files
    os.makedirs(model_dir, exist_ok=True)
    for filename in filenames:
        file_url = os.path.join(base_url, model_size, filename)
        file_path = os.path.join(model_dir, filename)
        print(file_url, file_path)
        download_file(file_url, file_path)
    
    # Add tf loading part
    tf_ckpt_path = tf.train.latest_checkpoint(model_dir)
    settings = json.load(open(os.path.join(model_dir, 'hparams.json')))
    params = load_gpt2_params_from_tf_ckpt(tf_ckpt_path, settings)
    return settings, params
