import numpy as np
import torch
import os
from torch import Tensor
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F

from model import DiffWave


class DiffWaveTrainer:
    def __init__(self, model: DiffWave, optimizer, dataset, model_dir, params, *args, **kwargs):
        if not os.path.exists(model_dir):
            os.makedirs(name=model_dir, exist_ok=True)

        self.model = model
        self.model_dir = model_dir
        self.dataset = dataset
        self.params = params
        self.optimizer = optimizer

        # TODO tune this for mixed precision training
        self.step = 0
        beta = np.array(self.params.noise_schedule)
        noise_level = np.cumprod(1 - beta)  # alpha
        self.loss_fn = nn.L1Loss()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.noise_level = torch.tensor(noise_level.astype(np.float32)).to(device=self.device)
        self.epochs = params.epochs

    def loss_function(self):
        pass

    def train_step(self, audio: Tensor, condition=None):
        audio = audio.to(device=self.device)
        if condition:
            condition = condition.to(self.device)

        N, T = audio.shape

        self.optimizer.zero_grad()

        t = torch.randint(low=0, high=len(self.params.noise_schedule), size=[N], device=self.device)
        noise_scale = self.noise_level[t].unsqueeze(1)
        noise_scale_sqrt = noise_scale ** 0.5
        noise = torch.rand_like(audio)
        noisy_audio = (noise_scale_sqrt * audio) + ((1. - noise_scale) ** 0.5) * noise
        predicted = self.model(noisy_audio, t, condition)
        loss = self.loss_fn(noise, predicted.squeeze(1))

        loss.backward()

        self.optimizer.step()

        return loss

    def train(self):
        for epoch in tqdm(self.epochs, desc="Epochs"):
            for i, data in tqdm(enumerate(self.dataset), desc=f"Epoch {epoch + 1}"):
                audio, condition = data
                loss = self.train_step(audio=audio, condition=condition)

        # write summary to the disk


if __name__ == "__main__":
    pass
