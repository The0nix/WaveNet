from pathlib import Path
from typing import Tuple, Sequence

import numpy as np
import torch
import torch.utils.data as torchdata
import torchaudio

import core.utils


class LJSPEECH(torchdata.Dataset):
    """
    Wrapper for torchaudio.datasets.SPEECHCOMMANDS with predefined keywords
    :param root: Path to the directory where the dataset is found or downloaded.
    :param transforms: audiomentations transform object for waveform transform
    :param download: Whether to download the dataset if it is not found at root path. (default: False)
    """
    def __init__(self, root: str, transforms, crop_size=None, n_mu_law=256, download: bool = False) -> None:
        root = Path(root)
        if download and not root.exists():
            root.mkdir()
        self.dataset = torchaudio.datasets.LJSPEECH(root=root, download=download)
        self.transforms = transforms
        self.crop_size = crop_size
        self.n_mu_law = n_mu_law

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str]:
        """
        :return: mu-law encoded waveform, transformed waveform
        """
        (waveform, sample_rate, transcript, normalized_transcript) = self.dataset[idx]
        if self.crop_size is not None:
            waveform = core.utils.random_crop_waveform(waveform, self.crop_size)
        waveform_transformed = self.transforms(samples=waveform, sample_rate=sample_rate)
        waveform_encoded = torchaudio.functional.mu_law_encoding(waveform, self.n_mu_law)
        return waveform_encoded, waveform_transformed  # May be not waveform already

    def __len__(self) -> int:
        return len(self.dataset)

    def get_transcript(self, idx: int) -> str:
        """
        Get normalized_transcript only from the dataset
        :param idx: object index
        :return: normalized_transcript keyword_id
        """
        fileid, transcript, normalized_transcript = self.dataset._walker[idx]
        return normalized_transcript


class RandomBySequenceLengthSampler(torchdata.Sampler):
    """
    Samples batches by bucketing them to similar lengths examples
    (Note: drops last batch)
    :param lengths: list of lengths of examples in dataset
    :param batch_size: batch size
    :param percentile: what percentile of lengths to include (e.g. 0.9 for 90% of smallest lengths)
    """
    def __init__(self, lengths: Sequence, batch_size, percentile=1.0):
        super().__init__(lengths)
        indices = np.argsort(lengths)
        indices = indices[:int(len(indices) * percentile)]
        indices = indices[:len(indices) - (len(indices) % batch_size)]
        self.batched_indices = indices.reshape(-1, batch_size)

    def __iter__(self):
        metaindices = np.random.permutation(len(self.batched_indices))
        return iter(self.batched_indices[metaindices])

    def __len__(self):
        return len(self.batched_indices)
