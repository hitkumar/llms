import torch
from tqdm import tqdm
from ddpm import DDPMSampler
import numpy as np

WIDTH = 512
HEIGHT = 512
LATENT_WIDTH = WIDTH // 8
LATENT_HEIGHT = HEIGHT // 8

def rescale(x, old_range, new_range, clamp=False):
    old_min, old_max = old_range
    new_min, new_max = new_range
    x -= old_min
    x *= (new_max - new_min) / (old_max - old_min)
    x += new_min
    if clamp:
        x = x.clamp(new_min, new_max)
    return x

def get_time_embedding(timestep):
    # (160, )
    freqs = torch.pow(10000, -torch.arange(start=0, end=160, dtype=torch.float32) / 160)
    # (1, 160)
    x = torch.tensor([timestep], dtype=torch.float32).unsqueeze(0) * freqs.unsqueeze(0)
    # (1, 320)
    return torch.cat([torch.cos(x), torch.sin(x)], dim=1)

def generate(
    prompt,
    uncond_prompt=None,
    input_image=None,
    strength=0.8,
    do_cfg=True,
    cfg_scale=7.5,
    sampler_name="ddpm",
    n_inference_steps=50,
    models={},
    seed=None,
    device=None,
    tokenizer=None
):
    with torch.no_grad():
        if not 0 < strength <= 1:
            raise ValueError("strength must be between 0 and 1")
        
        generator = torch.Generator(device=device)
        if seed is None:
            generator.seed()
        else:
            generator.manual_seed(seed)
        
        clip = models['clip']

        if do_cfg:
            tokens = tokenizer.batch_encode_plus(
                [prompt, uncond_prompt], padding="max_length", max_length=77
            ).input_ids
            cond_tokens = torch.tensor(tokens[0], device=device)
            uncond_tokens = torch.tensor(tokens[1], device=device)

            tokens_cat = torch.cat([cond_tokens.unsqueeze(0), uncond_tokens.unsqueeze(0)], dim=0)
            # (2 * B, seq_len, Dim) -> 2, 77, 768]
            context = clip(tokens_cat)
        else:
            tokens = tokenizer.batch_encode_plus(
                [prompt], padding="max_length", max_length=77
            ).input_ids
            # (1, seq_len)
            cond_tokens = torch.tensor(tokens, device=device)
            context = clip(cond_tokens)
        
        if sampler_name == "ddpm":
            sampler = DDPMSampler(generator)
            sampler.set_inference_timesteps(n_inference_steps)
        else:
            raise ValueError(f"Unknown sampler value {sampler_name}")
        
        # assuming batch size of 1
        latent_shape = (1, 4, LATENT_HEIGHT, LATENT_WIDTH)

        if input_image:
            encoder = models["encoder"]
            input_image_resize = input_image.resize((HEIGHT, WIDTH))
            # (H, W, C)
            input_image_tensor = torch.tensor(np.array(input_image_resize), dtype=torch.float32, device=device)
            # (1, H, W, C)
            input_image_tensor = rescale(input_image_tensor, (0, 255), (-1, 1)).unsqueeze(0)
            # (1, C, H, W)
            input_image_tensor = input_image_tensor.permute(0, 3, 1, 2)

            # (1, 4, H / 8, W / 8)
            encoder_noise = torch.randn(latent_shape, generator=generator, device=device)
            # (1, 4, H / 8, W / 8)
            latents = encoder(input_image_tensor, encoder_noise) # this is input to diffusion UNET

            sampler.set_strength(strength=strength)
            # Add the initial amount of noise to the input image
            latents = sampler.add_noise(latents, sampler.timesteps[0])
        else:
            latents = torch.randn(latent_shape, generator=generator, device=device)
        
        diffusion = models["diffusion"]
        timesteps = tqdm(sampler.timesteps)
        for i, timestep in enumerate(timesteps):
            # (1, 320)
            time_embedding = get_time_embedding(timestep)
            model_input = latents
            if do_cfg:
                model_input = model_input.repeat(2, 1, 1, 1)

            # this is the noise
            model_output = diffusion(model_input, context, time_embedding)
            if do_cfg:
                output_cond, output_uncond = model_output.chunk(2)
                model_output = cfg_scale * (output_cond - output_uncond) + output_uncond

            # remove the noise
            latents = sampler.step(timestep, latents, model_output)
        
        decoder = models["decoder"]
        # (B, 3, H, W)
        images = decoder(latents)
        images = rescale(images, (-1, 1), (0, 255), clamp=True)
        # (B, 3, H, W) -> (B, H, W, C)
        images = images.permute(0, 2, 3, 1)
        images = images.to("cpu", torch.uint8).numpy()
        return images[0]