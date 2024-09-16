from typing import Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import torch
from PIL import Image

IMAGENET_STANDARD_MEAN = [0.5, 0.5, 0.5]
IMAGENET_STANDARD_STD = [0.5, 0.5, 0.5]


def add_image_tokens_to_prompt(prefix_prompt, bos_token, image_seq_len, image_token):
    """TODO: From paper, \n should be tokenized separately, but it's not in HF tokenizer.
    Lets try this"""
    return f"{image_seq_len * image_token}{bos_token}{prefix_prompt}\n"


def rescale(
    image: np.ndarray, scale: float, dtype: np.dtype = np.float32
) -> np.ndarray:
    """Rescale the image by a given scale."""
    return (image * scale).astype(dtype)


def resize(
    image: Image,
    size: Tuple[int, int],
    resampling: Image.Resampling,
    reducing_gap: Optional[int] = None,
) -> np.ndarray:
    height, width = size
    return image.resize((width, height), resample=resampling, reducing_gap=reducing_gap)


def normalize(
    image: np.ndarray,
    mean: Union[float, Iterable[float]],
    std: Union[float, Iterable[float]],
) -> np.ndarray:
    mean = np.array(mean, dtype=image.dtype)
    std = np.array(std, dtype=image.dtype)
    return (image - mean) / std


def process_images(
    images: List[Image.Image],
    size: Tuple[int, int],
    resample: Image.Resampling = None,
    rescale_factor: float = None,
    image_mean: Optional[Union[float, Iterable[float]]] = None,
    image_std: Optional[Union[float, Iterable[float]]] = None,
) -> List[np.ndarray]:
    height, width = size
    images = [
        resize(image=image, size=(height, width), resampling=resample)
        for image in images
    ]
    images = [np.array(image) for image in images]
    # rescale images to be in [0, 1]
    images = [rescale(image=image, scale=rescale_factor) for image in images]
    # normalize images
    images = [
        normalize(image=image, mean=image_mean, std=image_std) for image in images
    ]
    # move the channel dimnension to the first dim so that output is [channel, height, width]
    images = [image.transpose(2, 0, 1) for image in images]
    return images


class PaliGemmaProcessor:
    IMAGE_TOKEN = "<image>"

    def __init__(self, tokenizer, num_image_tokens: int, image_size: int):
        super().__init__()
        self.image_seq_len = num_image_tokens
        self.image_size = image_size

        tokens_to_add = {"additional_special_tokens": [self.IMAGE_TOKEN]}
        tokenizer.add_special_tokens(tokens_to_add)
        EXTRA_TOKENS = [f"<loc{id:04d}>" for id in range(1024)]
        EXTRA_TOKENS += [f"<seg{id:03d}>" for id in range(128)]
        tokenizer.add_tokens(EXTRA_TOKENS)
        self.image_token_id = tokenizer.convert_tokens_to_ids(self.IMAGE_TOKEN)
        tokenizer.add_bos_token = False
        tokenizer.add_eos_token = False

        self.tokenizer = tokenizer

    def __call__(
        self,
        texts: List[str],
        images: List[Image.Image],
        padding: str = "longest",
        truncation: bool = True,
    ) -> dict:
        assert len(texts) == 1 and len(images) == 1
        # list of np array of shape [channel, height, width]
        pixel_values = process_images(
            images=images,
            size=(self.image_size, self.image_size),
            resample=Image.Resampling.BICUBIC,
            rescale_factor=1.0 / 255.0,
            image_mean=IMAGENET_STANDARD_MEAN,
            image_std=IMAGENET_STANDARD_STD,
        )
        pixel_values = np.stack(pixel_values, axis=0)
        # [N, C, H, W]
        pixel_values = torch.tensor(pixel_values)

        input_strings = [
            add_image_tokens_to_prompt(
                prefix_prompt=prompt,
                bos_token=self.tokenizer.bos_token,
                image_seq_len=self.image_seq_len,
                image_token=self.IMAGE_TOKEN,
            )
            for prompt in texts
        ]

        inputs = self.tokenizer(
            input_strings,
            padding=padding,
            truncation=truncation,
            return_tensors="pt",
        )

        return {"pixel_values": pixel_values, **inputs}
