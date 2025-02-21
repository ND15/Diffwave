from dataclasses import dataclass, field
from typing import List, Optional

from utilities.utils import cosine_schedule


@dataclass
class Params:
    batch_size: int = 4
    learning_rate: float = 2e-4
    max_grad_norm: Optional[float] = None
    epochs: int = 10
    time_steps: int = 50

    sample_rate: int = 22050
    n_mels: int = 80
    n_fft: int = 1024
    hop_samples: int = 256
    crop_mel_frames: int = 62

    # Model params
    residual_layers: int = 10
    residual_channels: int = 64
    dilation_cycle_length: int = 10
    condition: bool = False
    noise_schedule: List[float] = field(default_factory=lambda: cosine_schedule(time_steps=50))
    inference_noise_schedule: List[float] = field(default_factory=lambda: [0.0001, 0.001, 0.01, 0.05, 0.2, 0.5])

    audio_len: int = 16000 * 3
