import math

import torch


def cosine_schedule(time_steps, epsilon=0.008):
    steps = torch.arange(time_steps)
    f_t = torch.cos(((steps / time_steps + epsilon) / (1.0 + epsilon)) * (math.pi / 2)) ** 2
    alpha_t = f_t / f_t[0]
    beta_t = torch.clip(1.0 - alpha_t[1:] / alpha_t[:-1], 0.0, 0.999)
    return beta_t
