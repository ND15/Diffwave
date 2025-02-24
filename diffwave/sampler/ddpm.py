import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm


class DDPMSampler(nn.Module):
    def __init__(self, model, params):
        super().__init__()
        self.params = params
        self.device = "cuda" if torch.cuda.is_available() else "cpu"  # Set the device

        # Convert noise schedule to tensor and move to device
        beta = np.array(self.params.noise_schedule)
        self.register_buffer(name="betas", tensor=torch.tensor(beta, dtype=torch.float32).to(self.device))

        # Compute derived tensors and move to device
        alphas = 1.0 - self.betas
        alphas_cum_prod = torch.cumprod(alphas, dim=-1)

        self.model = model.to(self.device)  # Move model to device
        self.params = params

        # Register buffers and move to device
        self.register_buffer(name="alphas", tensor=alphas.to(self.device))
        self.register_buffer(name="alphas_cum_prod", tensor=alphas_cum_prod.to(self.device))
        self.register_buffer(name="sqrt_alphas_cum_prod", tensor=torch.sqrt(alphas_cum_prod).to(self.device))
        self.register_buffer(name="sqrt_one_minus_alphas_cum_prod",
                             tensor=torch.sqrt(1. - alphas_cum_prod).to(self.device))

    @torch.no_grad()
    def sample_one_step(self, x_t, time_step, condition=None):
        """
        Check the formula for calculating mean and variance in DDPM paper
        """
        x_t = x_t.to(self.device).float()
        time_step = time_step.to(self.device)
        if condition is not None:
            condition = condition.to(self.device).float()

            # Model prediction
        prediction = self.model(x_t, time_step, condition).squeeze(dim=1)
        noise = torch.randn_like(prediction).to(self.device).float()

        b, t = x_t.shape
        prediction = prediction[..., :t]

        # Gather alpha and beta values for the current time step
        alpha_t = self.alphas.gather(dim=-1, index=time_step).reshape(b, 1).to(self.device)
        beta_t = self.betas.gather(dim=-1, index=time_step).reshape(b, 1).to(self.device)

        alpha_t_cum_prod = self.alphas_cum_prod.gather(dim=-1, index=time_step).reshape(b, 1).to(self.device)
        sqrt_one_minus_alphas_cum_prod = self.sqrt_one_minus_alphas_cum_prod.gather(
            dim=-1, index=time_step
        ).reshape(b, 1).to(self.device)

        mean = (1. / torch.sqrt(alpha_t)) * (x_t - ((1. - alpha_t) / sqrt_one_minus_alphas_cum_prod) * prediction).to(
            self.device)

        if time_step.min() > 0:
            alpha_t_cum_prod_prev = self.alphas_cum_prod.gather(
                dim=-1, index=time_step - 1
            ).reshape(b, 1).to(self.device)
            std = torch.sqrt(beta_t * (1 - alpha_t_cum_prod_prev) / (1 - alpha_t_cum_prod)).to(self.device)
        else:
            std = torch.zeros_like(mean).to(self.device)

        return mean + std * noise

    @torch.no_grad()
    def sampling(self, n_samples, conditions=None):
        x_t = torch.randn((n_samples, self.params.audio_len), dtype=torch.float32).to(self.device)  # Ensure float32

        for i in tqdm(range(self.params.time_steps - 1, -1, -1), desc="Sampling"):
            t = torch.tensor([i for _ in range(n_samples)], dtype=torch.long).to(self.device)
            x_t = self.sample_one_step(x_t=x_t, time_step=t, condition=conditions)

        # x_t = (x_t + 1.) / 2.

        return x_t
