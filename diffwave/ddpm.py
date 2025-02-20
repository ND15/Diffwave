from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class DDPMSampler(nn.Module):
    def __init__(self, model: nn.Module, params):
        super().__init__()
        beta = np.array(self.params.noise_schedule)
        self.register_buffer(name="beta_t", tensor=torch.tensor(beta))

        alpha_t = 1.0 - self.beta_t
        alpha_t_bar = torch.cumprod(alpha_t, dim=0)
        alpha_t_bar_prev = F.pad(alpha_t_bar[:-1], (1, 0), value=1.0)

        self.register_buffer("coeff_1", 1.0 / torch.sqrt(alpha_t))
        self.register_buffer("coeff_2", self.coeff_1 * (1.0 - alpha_t) / torch.sqrt(1.0 - alpha_t_bar))
