import warnings
from typing import Union

import librosa
import numpy as np
import torch
import torch.nn as nn
import torchaudio


class MelSpectrogram(nn.Module):
    """
    torchaudio MelSpectrogram wrapper for audiomentations's Compose
    """
    def __init__(self, clip_min_value=1e-5, *args, **kwargs):
        super().__init__()
        self.transform = torchaudio.transforms.MelSpectrogram(**kwargs)
        self.clip_min_value = clip_min_value

        mel_basis = librosa.filters.mel(
            sr=kwargs["sample_rate"],
            n_fft=kwargs["n_fft"],
            n_mels=kwargs["n_mels"],
            fmin=kwargs["f_min"],
            fmax=kwargs["f_max"],
        ).T
        self.transform.mel_scale.fb.copy_(torch.tensor(mel_basis))

    def forward(self, samples: Union[np.ndarray, torch.Tensor], sample_rate: int) -> torch.Tensor:
        if not isinstance(samples, torch.Tensor):
            samples = torch.tensor(samples)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            samples = self.transform.forward(samples)
        samples.clamp_(min=self.clip_min_value)
        return samples


class Squeeze:
    """
    Transform to squeeze monochannel waveform
    """
    def __call__(self, samples: Union[np.ndarray, torch.Tensor], sample_rate: int):
        return samples.squeeze(0)


class ToNumpy:
    """
    Transform to make numpy array
    """
    def __call__(self, samples: Union[np.ndarray, torch.Tensor], sample_rate: int):
        return np.array(samples)


class LogTransform(nn.Module):
    """
    Transform for taking logarithm of mel spectrograms (or anything else)
    :param fill_value: value to substitute non-positive numbers with before applying log
    """
    def __init__(self, fill_value: float = 1e-5) -> None:
        super().__init__()
        self.fill_value = fill_value

    def __call__(self, samples: torch.Tensor, sample_rate: int):
        samples = samples.masked_fill((samples <= 0), self.fill_value)
        return torch.log(samples)
