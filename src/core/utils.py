import random
from typing import Sequence

import audiomentations as aud
import hydra
import numpy as np
import sklearn
import torch
import torch.nn as nn
import torch.utils.data as torchdata
from omegaconf import DictConfig


def fix_seeds(seed=1337):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True


def get_transforms(transforms: DictConfig):
    """
    get all necessary transforms from config
    :param transforms: transforms from config
    :return: transforms composed into aud.Compose
    """
    if transforms is None:
        return None
    return aud.Compose([
        hydra.utils.instantiate(transform)
        for transform in transforms
    ])


def random_crop_waveform(waveform: torch.Tensor, crop_size: int):
    """
    Randomly crop wavefrom to crop_size
    :param waveform: waveform to crop
    :param crop_size: size to crop to
    :return:
    """
    start = random.randrange(waveform.shape[-1] - crop_size)
    return waveform[:,start: start + crop_size]



def get_split(dataset: torchdata.Dataset, train_size: float, random_state: int):
    """
    Get train and test indices for dataset
    :param dataset: torch.Dataset (or any object with length)
    :param train_size: fraction of indices to use for training
    :param random_state: random state for dataset
    :return:
    """
    idxs = np.arange(len(dataset))
    train_idx, test_idx = sklearn.model_selection.train_test_split(idxs, train_size=train_size,
                                                                   random_state=random_state)
    return train_idx, test_idx


class PadCollator:
    def __init__(self, spectrogram_padding_value=-999., waveform_padding_value=0):
        self.spectrogram_padding_value = spectrogram_padding_value
        self.waveform_padding_value = waveform_padding_value

    def __call__(self, batch):
        waveforms, spectrograms = zip(*batch)

        waveform_lengths = torch.tensor([t.shape[1] for t in waveforms])
        spectrogram_lengths = torch.tensor([s.shape[1] for s in spectrograms])
        waveforms = nn.utils.rnn.pad_sequence(
            [s.transpose(0, 1) for s in waveforms],
            batch_first=True, padding_value=self.waveform_padding_value
        ).transpose(1, 2)
        spectrograms = nn.utils.rnn.pad_sequence(
            [s.transpose(0, 1) for s in spectrograms],
            batch_first=True, padding_value=self.spectrogram_padding_value
        ).transpose(1, 2)

        return waveforms, spectrograms, waveform_lengths, spectrogram_lengths


def lengths_to_mask(lengths: torch.Tensor) -> torch.Tensor:
    """
    Creates masking tensor from a list of lengths
    :param lengths: tensor of lengths
    :return: tensor of shape (batch_size, max_length) with mask (Trues and Falses)
    """
    return torch.arange(max(lengths), device=lengths.device)[None, :] < lengths[:, None]
