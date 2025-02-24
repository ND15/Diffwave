from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np


@dataclass
class Params:
    batch_size: int = 4
    learning_rate: float = 2e-4
    max_grad_norm: Optional[float] = None
    epochs: int = 100
    time_steps: int = 50

    sample_rate: int = 16000
    n_mels: int = 80
    n_fft: int = 1024
    hop_samples: int = 256
    crop_mel_frames: int = 62

    # Model params
    residual_layers: int = 30
    residual_channels: int = 64
    dilation_cycle_length: int = 10
    condition: bool = False
    # noise_schedule: List[float] = field(default_factory=lambda: cosine_schedule(time_steps=1000))
    noise_schedule: List[float] = field(default_factory=lambda: np.linspace(1e-4, 0.05, 50).tolist())
    inference_noise_schedule: List[float] = field(default_factory=lambda: [0.0001, 0.001, 0.01, 0.05, 0.2, 0.5])
    audio_len: int = 16000 * 3
