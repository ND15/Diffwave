import os

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
    def __init__(self, model: DiffWave, sampler, optimizer, dataset, model_dir, params):
        if not os.path.exists(model_dir):
            os.makedirs(name=model_dir, exist_ok=True)

        self.model = model
        self.model_dir = model_dir
        self.dataset = dataset
        self.params = params
        self.optimizer = optimizer
        self.sampler = sampler

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.scaler = torch.cuda.amp.GradScaler()
        self.step = 0
        betas = torch.tensor(self.params.noise_schedule)
        alphas = 1.0 - betas
        self.alphas = alphas.to(self.device)
        self.alphas_cum_prod = torch.cumprod(alphas, dim=-1).to(self.device)
        self.sqrt_alphas_cum_prod = torch.sqrt(self.alphas_cum_prod).to(self.device)
        self.sqrt_one_minus_alphas_cum_prod = torch.sqrt(1. - self.alphas_cum_prod).to(self.device)
        self.loss_fn = nn.MSELoss()
        self.epochs = params.epochs

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

        with torch.cuda.amp.autocast():
            predicted = self.model(noisy_audio, t, condition)
            loss = self.loss_fn(noise, predicted.squeeze(1))

        self.scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.scaler.step(self.optimizer)
        self.scaler.update()

        return loss

    def train(self, resume_checkpoint: str = f"train/checkpoint_epoch_{16}.pt"):
        start_epoch = 0
        if resume_checkpoint:
            start_epoch = self.load_checkpoint(resume_checkpoint)

        # printc(f"[INFO] Dataset Length: {len(self.dataset)}")

        for epoch in tqdm(range(start_epoch, self.epochs), desc="Epochs"):
            epoch_loss = 0
            for i, data in tqdm(enumerate(self.dataset), desc=f"Epoch {epoch + 1}"):
                if len(data) == 1:
                    audio = data[0]
                    condition = None
                else:
                    audio, condition = data

                loss = self.train_step(audio=audio, condition=condition)
                epoch_loss += loss.detach().cpu()

            avg_loss = epoch_loss / len(self.dataset)
            print(f"Epoch {epoch + 1:03d} | Avg Loss: {avg_loss:.4f}")

            # Save checkpoint every 2 epochs
            if (epoch + 1) % 2 == 0:
                self.save_checkpoint(epoch + 1)
                # samples = self.sampler.sampling(n_samples=10)
                # for j, sample in enumerate(samples):
                #     if not os.path.exists(self.model_dir + f"/epoch_{epoch + 1}"):
                #         os.makedirs(self.model_dir + f"/epoch_{epoch + 1}")
                #
                #     soundfile.write(f"{self.model_dir}/epoch_{epoch + 1}/sample_{j}.wav",
                #                     data=sample.detach().cpu().numpy(),
                #                     samplerate=16000)

    def generate(self, model_path: str = f"train/checkpoint_epoch_{16}.pt", num_samples: int = 10):
        self.load_checkpoint(model_path)
        samples = self.sampler.sampling(n_samples=num_samples)
        for j, sample in enumerate(samples):
            if not os.path.exists(self.model_dir + f"/generated"):
                os.makedirs(self.model_dir + f"/generated")
            soundfile.write(f"{self.model_dir}/generated/sample_{j}.wav",
                            data=sample.detach().cpu().numpy(),
                            samplerate=16000)

    def save_checkpoint(self, epoch: int):
        checkpoint = {
            'epoch': epoch,
            'step': self.step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scaler_state_dict': self.scaler.state_dict(),
        }
        checkpoint_path = os.path.join(self.model_dir, f"checkpoint_epoch_{epoch}.pt")
        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved at {checkpoint_path}")

    def load_checkpoint(self, checkpoint_path: str):
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        self.step = checkpoint['step']
        epoch = checkpoint['epoch']
        print(f"Checkpoint loaded from {checkpoint_path}. Resuming from epoch {epoch}, step {self.step}")
        return epoch


if __name__ == "__main__":
    params = Params()
    audio_dataset = UnconditionalAudioDataset("../../../../../Data/audio/audio/", num_segments=2, segment_length=3)
    dataloader = DataLoader(audio_dataset, batch_size=16, shuffle=True)
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
    # trainer.generate()
