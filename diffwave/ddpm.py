import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm


class DDPMSampler(nn.Module):
    def __init__(self, model: nn.Module, params):
        super().__init__()
        beta = np.array(self.params.noise_schedule)
        self.register_buffer(name="betas", tensor=torch.tensor(beta))

        alphas = 1.0 - self.betas
        alphas_cum_prod = torch.cumprod(self.beta_t, dim=-1)

        self.model = model
        self.params = params

        self.register_buffer(name="betas", tensor=self.betas)
        self.register_buffer(name="alphas", tensor=alphas)
        self.register_buffer(name="alphas_cum_prod", tensor=alphas_cum_prod)
        self.register_buffer(name="sqrt_alphas_cum_prod", tensor=torch.sqrt(alphas_cum_prod))
        self.register_buffer(name="sqrt_one_minus_alphas_cum_prod", tensor=torch.sqrt(1. - alphas_cum_prod))

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    @torch.no_grad()
    def sample_one_step(self, x_t, time_step, condition=None):
        """
        Check the formula for calculating mean and variance in DDPM paper
        """
        prediction = self.model(x_t, time_step, condition)
        noise = torch.randn_like(prediction)

        b, c, t = x_t.shape
        prediction = prediction[..., :t]

        alpha_t = self.alphas.gather(dim=-1, index=time_step).reshape(b, 1, 1)
        beta_t = self.betas.gather(dim=-1, index=time_step).reshape(b, 1, 1)

        alpha_t_cum_prod = self.alphas_cum_prod.gather(dim=-1, index=time_step).reshape(b, 1, 1)
        sqrt_one_minus_alphas_cum_prod = self.sqrt_one_minus_alphas_cum_prod.gather(dim=-1,
                                                                                    index=time_step).reshape(b, 1, 1)

        mean = (1. / torch.sqrt(alpha_t)) * (x_t - ((1. - alpha_t) / sqrt_one_minus_alphas_cum_prod) * prediction)

        if time_step > 0:
            alpha_t_cum_prod_prev = self.alphas_cum_prod.gather(dim=-1,
                                                                index=time_step - 1).reshape(b, 1, 1)
            std = torch.sqrt(beta_t * (1 - alpha_t_cum_prod_prev) / (1 - alpha_t_cum_prod))
        else:
            std = 0.0

        return mean + std * noise

    @torch.no_grad()
    def sampling(self, n_samples, conditions=None):
        x_t = torch.randn((n_samples, 1, self.params.audio_length))
        for i in tqdm(range(self.params.time_steps)):
            t = torch.tensor([i for _ in range(n_samples)]).to(self.device)
            x_t = self.sample_one_step(x_t=x_t, time_step=t, condition=conditions)

        x_t = (x_t + 1.) / 2.

        return x_t
