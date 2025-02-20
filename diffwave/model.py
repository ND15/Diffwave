import math
import torch.nn as nn
import torch.nn.functional as F
from layers import DiffEmbedding, SpectrogramUpsampler, ResidualBlock


class DiffWave(nn.Module):
    def __init__(self, params, **kwargs):
        super().__init__(**kwargs)
        self.params = params
        self.input_projection = nn.Conv1d(in_channels=1, out_channels=params.residual_channels, kernel_size=1)
        self.diffusion_embedding = DiffEmbedding(len(params.noise_schedule))
        if self.params.condition:
            self.condition_layer = SpectrogramUpsampler(params.n_mels)  # need to change this for text
        else:
            self.condition_layer = None

        self.residual_layers = nn.ModuleList([
            ResidualBlock(n=params.n_mels, residual_channels=params.residual_channels,
                          dilation_rate=2 ** (i % params.dilation_cycle_length),
                          condition=params.condition)
            for i in range(params.residual_layers)
        ])

        self.skip_projection = nn.Conv1d(in_channels=params.residual_channels,
                                         out_channels=params.residual_channels,
                                         kernel_size=1)
        self.output_projection = nn.Conv1d(in_channels=params.residual_channels,
                                           out_channels=1,
                                           kernel_size=1)

        nn.init.zeros_(self.output_projection.weight)

    def forward(self, inputs, diffusion_step, condition=None):
        x = inputs.unsqueeze(-1)
        x = self.input_projection(x)
        x = F.relu(x)

        diffusion_step = self.diffusion_embedding(diffusion_step)

        if self.condition_layer:
            condition = self.condition_layer(condition)

        skip = None
        for layer in self.residual_layers:
            x, skip_connection = layer(x, diffusion_step, condition)
            skip = skip_connection if skip is None else skip_connection + skip

        x = skip / math.sqrt(len(self.residual_layers))
        x = self.skip_projection(x)
        x = F.relu(x)
        x = self.output_projection(x)
        return x
