import os
import sys

sys.path.append(os.getcwd())
print(sys.path)
import soundfile
import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import DiffWave
from sampler.ddpm import DDPMSampler
from utilities.dataset import UnconditionalAudioDataset
from utilities.params import Params


class DiffWaveTrainer:
    def __init__(self, model: DiffWave,
                 sampler,
                 optimizer,
                 dataset,
                 model_dir,
                 params):
        if not os.path.exists(model_dir):
            os.makedirs(name=model_dir, exist_ok=True)

        self.model = model
        self.model_dir = model_dir
        self.dataset = dataset
        self.params = params
        self.optimizer = optimizer
        self.sampler = sampler

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # TODO tune this for mixed precision training
        self.step = 0
        betas = torch.tensor(self.params.noise_schedule)
        alphas = 1.0 - betas
        self.alphas = alphas.to(self.device)
        self.alphas_cum_prod = torch.cumprod(alphas, dim=-1)
        self.alphas_cum_prod = self.alphas_cum_prod.to(self.device)
        self.sqrt_alphas_cum_prod = torch.sqrt(self.alphas_cum_prod).to(self.device)
        self.sqrt_one_minus_alphas_cum_prod = torch.sqrt(1. - self.alphas_cum_prod).to(self.device)
        self.loss_fn = nn.MSELoss()
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
        noise = torch.randn_like(audio)
        noisy_audio = (self.sqrt_alphas_cum_prod.gather(dim=-1, index=t).reshape(audio.shape[0], 1) *
                       audio + self.sqrt_one_minus_alphas_cum_prod.gather(dim=-1, index=t).reshape(
                    noise.shape[0], 1) * noise)

        predicted = self.model(noisy_audio, t, condition)
        loss = self.loss_fn(noise, predicted.squeeze(1))

        loss.backward()

        self.optimizer.step()

        return loss

    def train(self):
        for epoch in tqdm(range(self.epochs), desc="Epochs"):
            epoch_loss = 0
            for i, data in tqdm(enumerate(self.dataset), desc=f"Epoch {epoch + 1}"):
                if len(data) == 1:
                    audio = data[0]
                    condition = None
                else:
                    audio, condition = data

                loss = self.train_step(audio=audio, condition=condition)
                epoch_loss += loss.detach().cpu()
                if (i + 1) % 100 == 0:
                    print(f"Epoch {epoch + 1:03d} | Loss: {epoch_loss / i:.4f}")
                    samples = self.sampler.sampling(n_samples=10)
                    for j, sample in enumerate(samples):
                        if not os.path.exists(self.model_dir + f"/epoch_{i + 1}_iter_{i}"):
                            os.makedirs(self.model_dir + f"/epoch_{i + 1}_iter_{i}")

                        soundfile.write(f"{self.model_dir}/epoch_{i + 1}_iter_{i}/sample_{j}.wav",
                                        data=sample.detach().cpu().numpy(),
                                        samplerate=16000)

            avg_loss = epoch_loss / len(self.dataset)
            print(f"Epoch {epoch + 1:03d} | Loss: {avg_loss:.4f}")
            samples = self.sampler.sampling(n_samples=10)
            for j, sample in enumerate(samples):
                if not os.path.exists(self.model_dir + f"/epoch_{epoch + 1}"):
                    os.makedirs(self.model_dir + f"/epoch_{epoch + 1}")

                soundfile.write(f"{self.model_dir}/epoch_{epoch + 1}/sample_{j}.wav",
                                data=sample.detach().cpu().numpy(),
                                samplerate=16000)


if __name__ == "__main__":
    params = Params()
    audio_dataset = UnconditionalAudioDataset("../../../../../Data/audio/audio/", num_segments=3)
    dataloader = DataLoader(audio_dataset, batch_size=4, shuffle=True)
    model = DiffWave(params=params).to(device="cuda")
    sampler = DDPMSampler(model=model, params=params)
    optimizer = torch.optim.Adam(model.parameters())
    trainer = DiffWaveTrainer(model=model,
                              sampler=sampler,
                              dataset=dataloader,
                              optimizer=optimizer,
                              model_dir="train",
                              params=params)
    trainer.train()
