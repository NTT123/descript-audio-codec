import os
import math
import warnings
from pathlib import Path

import argbind
import numpy as np
import torch
from audiotools import AudioSignal
from audiotools.core import util
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

from dac.utils import load_model

class AudioDataset(Dataset):
    def __init__(self, wav_dir):
        self.wav_files = [os.path.join(wav_dir, f) for f in os.listdir(wav_dir) if f.endswith('.wav')]

    def __len__(self):
        return len(self.wav_files)

    def __getitem__(self, idx):
        return self.wav_files[idx], AudioSignal(self.wav_files[idx])


warnings.filterwarnings("ignore", category=UserWarning)


@argbind.bind(group="encode", positional=True, without_prefix=True)
@torch.no_grad()
def encode(
    input: str,
    output: str = "",
    weights_path: str = "",
    model_tag: str = "latest",
    model_bitrate: str = "8kbps",
    n_quantizers: int = None,
    device: str = "cuda",
    model_type: str = "44khz",
    win_duration: float = 5.0,
    verbose: bool = False,
):
    """Encode audio files in input path to .dac format.

    Parameters
    ----------
    input : str
        Path to input audio file or directory
    output : str, optional
        Path to output directory, by default "". If `input` is a directory, the directory sub-tree relative to `input` is re-created in `output`.
    weights_path : str, optional
        Path to weights file, by default "". If not specified, the weights file will be downloaded from the internet using the
        model_tag and model_type.
    model_tag : str, optional
        Tag of the model to use, by default "latest". Ignored if `weights_path` is specified.
    model_bitrate: str
        Bitrate of the model. Must be one of "8kbps", or "16kbps". Defaults to "8kbps".
    n_quantizers : int, optional
        Number of quantizers to use, by default None. If not specified, all the quantizers will be used and the model will compress at maximum bitrate.
    device : str, optional
        Device to use, by default "cuda"
    model_type : str, optional
        The type of model to use. Must be one of "44khz", "24khz", or "16khz". Defaults to "44khz". Ignored if `weights_path` is specified.
    """
    generator = load_model(
        model_type=model_type,
        model_bitrate=model_bitrate,
        tag=model_tag,
        load_path=weights_path,
    )
    generator.to(device)
    generator.eval()
    kwargs = {"n_quantizers": n_quantizers}

    def collate_fn(batch):
        return batch[0]

    input = Path(input)
    dataset = AudioDataset(input)
    dataloader = DataLoader(dataset, batch_size=1, num_workers=8, shuffle=False, collate_fn=collate_fn)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)

    for file_path, signal in tqdm(dataloader, desc="Encoding files"):
        signal = signal.to(device)
        # Encode audio to .dac format
        artifact = generator.compress(signal, win_duration, verbose=verbose, **kwargs)

        # Compute output path
        file_path = Path(file_path)
        relative_path = file_path.relative_to(input)
        output_dir = output / relative_path.parent
        if not relative_path.name:
            output_dir = output
            relative_path = file_path
        output_name = relative_path.with_suffix(".dac").name
        output_path = output_dir / output_name
        output_path.parent.mkdir(parents=True, exist_ok=True)

        artifact.save(output_path)


if __name__ == "__main__":
    args = argbind.parse_args()
    with argbind.scope(args):
        encode()
