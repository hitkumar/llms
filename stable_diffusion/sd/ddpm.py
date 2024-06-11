import torch

class DDPMSampler:
    def __init__(self, generator: torch.Generator, num_training_steps=1000, beta_start=0.00085, beta_end=0.0120):
        self.betas = torch.linspace(beta_start ** 0.5, beta_end ** 0.5, num_training_steps, dtype=torch.float32) ** 2
        self.alphas = 1 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.one = torch.tensor(1.0)

        self.generator = generator
        self.num_train_timesteps = num_training_steps
        self.timesteps = torch.arange(num_training_steps - 1, -1, -1)
    
    def set_inference_timesteps(self, num_inference_steps=50):
        self.num_inference_steps = num_inference_steps
        step_ratio = self.num_train_timesteps // self.num_inference_steps
        self.step_ratio = step_ratio
        # In decreasing order 980, 960, 940 ....
        self.timesteps = torch.arange(num_inference_steps - 1, -1, -1) * step_ratio
    
    def _get_previous_timestep(self, timestep: int) -> int:
        prev_t = timestep - self.step_ratio
        return prev_t
    
    def add_noise(self, original_samples: torch.tensor, timesteps: torch.tensor):
        sqrt_alpha_prod = self.alphas_cumprod[timesteps] ** 0.5
        # TODO: Is this needed, verify the shapes
        sqrt_alpha_prod = sqrt_alpha_prod.flatten()
        while len(sqrt_alpha_prod.shape) < len(original_samples.shape):
            sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)
        
        variance = (1 - self.alphas_cumprod[timesteps]) ** 0.5
        variance = variance.flatten()
        while len(variance.shape) < len(original_samples.shape):
            variance = variance.unsqueeze(-1)
        
        # Like in eq(4) of DDPm paper, q(x_t/x_0) can be obtained
        noise = torch.randn(original_samples.shape, generator=self.generator, device=original_samples.device, dtype=original_samples.dtype)
        noisy_samples = sqrt_alpha_prod * original_samples + variance * noise
        return noisy_samples
    
    def set_strength(self, strength=1):
        '''
        More noise: output will be further from input
        less noise: outpout will be closer to input image
        '''
        start_step = self.num_inference_steps - int(self.num_inference_steps * strength)
        self.timesteps = self.timesteps[start_step:]
        self.start_step = start_step
    
    def step(self, t: int, latents: torch.tensor, model_output: torch.tensor):
        '''
        Remove noise from the latents and get latent at timestep (t-1)
        latents is x(t) at timestep t, model_output is the predicted noise
        '''
        prev_t = self._get_previous_timestep(t)

        # We use formula 6 and 7 of the paper, calculate alphas and beta first
        alpha_prod_t = self.alphas_cumprod[t]
        alpha_prod_t_prev = self.alphas_cumprod[prev_t] if prev_t >= 0 else self.one
        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev
        current_alpha_t = alpha_prod_t / alpha_prod_t_prev
        current_beta_t = 1 - current_alpha_t

        # Now calculate x_0
        pred_original_sample = (latents - (beta_prod_t ** 0.5) * model_output) / (alpha_prod_t ** 0.5)
        predicted_original_sample_coeff = ((alpha_prod_t_prev ** 0.5) * current_beta_t) / beta_prod_t
        current_sample_coeff = (current_alpha_t ** (0.5) * beta_prod_t_prev) / beta_prod_t

        prev_sample_mean = predicted_original_sample_coeff * pred_original_sample + current_sample_coeff * latents

        # variance
        variance = 0
        if t > 0:
            noise = torch.randn(model_output.shape, generator=self.generator, device=model_output.device, dtype=model_output.dtype)
            variance = (1 - alpha_prod_t_prev) / (1 - alpha_prod_t) * current_beta_t
            variance = torch.clamp(variance, min=1e-20)
            variance = variance ** 0.5
        
        pred_prev_sample = pred_prev_sample + variance * noise
        return pred_prev_sample