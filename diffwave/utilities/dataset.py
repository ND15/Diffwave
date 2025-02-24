import glob

import librosa
import numpy as np
import torch
from torch.utils import data


class UnconditionalAudioDataset(data.Dataset):
    def __init__(self, path_to_data: str,
                 sample_rate: int = 16000,
                 segment_length: int = 1,
                 num_segments: int = 1):
        self.filenames = glob.glob(path_to_data + "**/*.wav", recursive=True)
        self.filenames = self._filter_files_by_duration(self.filenames, min_duration=4.0)
        self.sample_rate = sample_rate
        self.segment_length = segment_length
        self.segment_samples = sample_rate * segment_length
        self.num_segments = num_segments

    @staticmethod
    def _filter_files_by_duration(filenames, min_duration=4.0):
        filtered_files = []
        for file in filenames:
            try:
                duration = librosa.get_duration(filename=file)
                if duration > min_duration:
                    filtered_files.append(file)
            except Exception as e:
                print(f"Error checking duration of file {file}: {e}")
        return filtered_files

    def __len__(self):
        return len(self.filenames) * self.num_segments

    def __getitem__(self, index):
        file_index = index // self.num_segments
        segment_index = index % self.num_segments

        try:
            audio, sr = librosa.load(self.filenames[file_index], sr=self.sample_rate)
        except Exception as e:
            print(f"Error loading file {self.filenames[file_index]}: {e}")
            audio = np.zeros(self.segment_samples)

        if len(audio) < self.segment_samples:
            padding = self.segment_samples - len(audio)
            audio = np.pad(audio, (0, padding), mode='constant')
        else:
            audio = audio[:self.segment_samples]

        start = segment_index * self.segment_samples
        end = start + self.segment_samples
        segment = audio[start:end]

        if len(segment) < self.segment_samples:
            padding = self.segment_samples - len(segment)
            segment = np.pad(segment, (0, padding), mode='constant')

        segment = torch.from_numpy(segment).float()

        return [segment]


if __name__ == "__main__":
    dataset = UnconditionalAudioDataset(path_to_data="../../../../../Data/audio/audio/", num_segments=5)

    segment = dataset[0]
    print(f"Segment shape: {segment[0].shape}")  # Should be (segment_samples,)

    from torch.utils.data import DataLoader

    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
    print(f"Number of batches: {len(dataloader)}")

    for batch in dataloader:
        print(f"Batch shape: {batch[0].shape}")  # Should be (batch_size, segment_samples)
        break
