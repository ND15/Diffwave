import glob
import math
import os
import subprocess
from multiprocessing import Pool

import matplotlib.pyplot as plt
import numpy as np
import torch


def cosine_schedule(time_steps, epsilon=0.008):
    """
    Generates a cosine schedule as specified in the DDPM paper
    """
    steps = torch.arange(start=0, end=time_steps + 1)
    f_t = torch.cos(((steps / time_steps + epsilon) / (1.0 + epsilon)) * (math.pi / 2)) ** 2
    beta_t = torch.clip(1.0 - f_t[1:] / f_t[:-1], 0.0, 0.999)
    return beta_t


def convert_mp3_to_wav(file_name):
    """
    Convert a single .mp3 file to .wav using ffmpeg.
    """
    filename, ext = os.path.splitext(file_name)
    try:
        # Convert .mp3 to .wav
        subprocess.call(
            ["ffmpeg", "-i", file_name, f"{filename}.wav"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT
        )
        # Remove the original .mp3 file
        os.remove(file_name)
        print(f"Converted and removed: {file_name}")
    except Exception as e:
        print(f"[ERROR] Failed to process {file_name}: {e}")


def mp3_to_wav(path_to_data: str):
    """
    Convert all .mp3 files in the given directory to .wav using parallel processing.
    """
    # Get all .mp3 files in the directory and its subdirectories
    file_names = glob.glob(os.path.join(path_to_data, "**/*.mp3"), recursive=True)

    # Determine the number of CPU cores available
    num_cores = os.cpu_count()
    print(num_cores)

    # Use multiprocessing to parallelize the conversion
    with Pool(processes=num_cores) as pool:
        pool.map(convert_mp3_to_wav, file_names)


if __name__ == "__main__":
    # mp3_to_wav("../../../../../Data/audio/audio/")
    x = np.linspace(1e-4, 0.05, 50)
    plt.plot(x)
    plt.show()
