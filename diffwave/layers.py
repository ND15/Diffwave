import torch
import torch.nn as nn
import torch.nn.functional as F


class DiffEmbedding(nn.Module):
    def __init__(self, max_steps, **kwargs):
        """
        Embedding layer for creating embedding for each diffusion timestep
        """
        super().__init__(**kwargs)
        self.max_steps = max_steps
        self.register_buffer(name="embedding", tensor=self._build_embeddings(),
                             persistent=False)
        self.linear_projection = nn.Sequential(
            nn.Linear(in_features=128, out_features=512),
            nn.SiLU(),
            nn.Linear(in_features=512, out_features=512),
            nn.SiLU(),
        )

    def forward(self, diffusion_step):
        if diffusion_step.dtype in [torch.int32, torch.int64]:
            x = self.embedding[diffusion_step]
        else:
            x = self._lerp_embedding(time_step=diffusion_step)

        x = self.linear_projection(x)
        return x

    def _lerp_embedding(self, time_step):
        """
        This embedding is for the fast sampling, where t can be a float.
        """
        low_idx = torch.floor(time_step).long()
        high_idx = torch.ceil(time_step).long()
        low = self.embedding[low_idx]
        high = self.embedding[high_idx]
        return low + (high - low) * (time_step - low_idx)

    def _build_embeddings(self):
        steps = torch.arange(self.max_steps).unsqueeze(1)
        dims = torch.arange(64).unsqueeze(0)
        embedding_matrix = steps * 10 ** (dims * 4.0 / 63.0)
        embedding_matrix = torch.cat((torch.sin(embedding_matrix),
                                      torch.cos(embedding_matrix)), dim=1)
        return embedding_matrix

    def _build_relative_embeddings(self):
        # embedding_matrix = torch.ra\ndn((self.max_steps, 64))
        pass


class SpectrogramUpsampler(nn.Module):
    def __init__(self, n_mels, **kwargs):
        super().__init__(**kwargs)
        self.conv_1 = nn.ConvTranspose2d(1, 1, (3, 32), stride=(1, 16), padding=(1, 8))
        self.conv_2 = nn.ConvTranspose2d(1, 1, (3, 32), stride=(1, 16), padding=(1, 8))

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.conv_1(x)
        x = F.leaky_relu(x, 0.4)
        x = self.conv_2(x)
        x = F.leaky_relu(x, 0.4)
        x = x.squeeze(1)
        return x


class ResidualBlock(nn.Module):
    def __init__(self, n, residual_channels, dilation_rate, condition=False, **kwargs):
        """
        """
        super().__init__(**kwargs)
        self.dilated_conv = nn.Conv1d(in_channels=residual_channels, out_channels=2 * residual_channels,
                                      dilation=dilation_rate, kernel_size=3,
                                      padding=dilation_rate)
        self.diffusion_projection = nn.Linear(in_features=512, out_features=residual_channels)

        if condition:
            self.conditioner_projection = nn.Conv1d(in_channels=n, out_channels=2 * residual_channels, kernel_size=1)
        else:
            self.conditioner_projection = None

        self.output_projection = nn.Conv1d(2 * residual_channels, residual_channels, kernel_size=1)
        self.skip_projection = nn.Conv1d(2 * residual_channels, residual_channels, kernel_size=1)

    def forward(self, x, diffusion_step, conditioner=None):
        # assertion
        diffusion_step = self.diffusion_projection(diffusion_step).unsqueeze(-1)
        y = x + diffusion_step

        if self.conditioner_projection is None:
            y = self.dilated_conv(y)

        else:
            conditioner = self.conditioner_projection(conditioner)
            y = self.dilated_conv(y) + conditioner

        gate = torch.sigmoid(y) * torch.tanh(y)

        y = self.output_projection(gate)

        skip = self.skip_projection(gate)

        return y, skip


if __name__ == "__main__":
    x = DiffEmbedding(max_steps=1000)
    y = SpectrogramUpsampler(80)
    residual = ResidualBlock(10, residual_channels=16, dilation_rate=3, condition=False)
    tensor = torch.randn((4, 16, 128))
    step = x(torch.randint(0, 1, (4,)))
    print(residual(tensor, step)[0].shape)
