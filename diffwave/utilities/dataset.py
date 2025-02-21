import glob

import librosa
import numpy as np
import torch
from torch.utils import data


class UnconditionalAudioDataset(data.Dataset):
    def __init__(self, path_to_data: str,
                 sample_rate: int = 16000,
                 segment_length: int = 3,
                 num_segments: int = 5):
        self.filenames = glob.glob(path_to_data + "**/*.wav", recursive=True)
        self.sample_rate = sample_rate
        self.segment_length = segment_length
        self.segment_samples = sample_rate * segment_length
        self.num_segments = num_segments

    def __len__(self):
        return len(self.filenames) * self.num_segments

    def __getitem__(self, index):
        file_index = index // self.num_segments
        segment_index = index % self.num_segments

        audio, sr = librosa.load(self.filenames[file_index], sr=self.sample_rate)

        if len(audio) < self.segment_samples * self.num_segments:
            padding = self.segment_samples * self.num_segments - len(audio)
            audio = np.pad(audio, (0, padding), mode='constant')

        start = segment_index * self.segment_samples
        end = start + self.segment_samples
        segment = audio[start:end]
        segment = torch.from_numpy(segment).float()

        return [segment]


if __name__ == "__main__":
    dataset = UnconditionalAudioDataset(path_to_data="../../../../../Data/audio/audio/", num_segments=5)

    # Access the 0th index
    segment = dataset[0]

    # Use with DataLoader
    from torch.utils.data import DataLoader

    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
    print(len(dataloader))
