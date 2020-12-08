from typing import Sequence, List, Tuple, Optional

import einops as eos
import einops.layers.torch as teos
import pytorch_lightning as pl
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchaudio
import wandb

import core.utils


class CausalConv1dXavier(torch.nn.Conv1d):
    """
    Causal convlution implementation from https://github.com/pytorch/pytorch/issues/1333#issuecomment-400338207
    :param gain_nonlinearity: what nonlinearity stands after the layer ("linear", "tanh", "sigmoid", etc.)
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 dilation=1, groups=1, bias=True, gain_nonlinearity: str = "linear"):
        self.__padding = (kernel_size - 1) * dilation
        super().__init__(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                         padding=self.__padding, dilation=dilation, groups=groups, bias=bias)
        nn.init.xavier_uniform_(self.weight, gain=nn.init.calculate_gain(gain_nonlinearity))

    def forward(self, input):
        result = super().forward(input)
        if self.__padding != 0:
            return result[:, :, :-self.__padding]
        return result


class Conv1dXavier(nn.Conv1d):
    """
    nn.Conv1d layer with xavier initialization
    :param gain_nonlinearity: what nonlinearity stands after the layer ("linear", "tanh", "sigmoid", etc.)
    """
    def __init__(self, *args, gain_nonlinearity: str = "linear", **kwargs):
        super().__init__(*args, **kwargs)
        nn.init.xavier_uniform_(self.weight, gain=nn.init.calculate_gain(gain_nonlinearity))


class ResidualBlock(nn.Module):
    def __init__(self, conv_channels, dilation):
        super().__init__()
        self.dilated_conv = CausalConv1dXavier(conv_channels, conv_channels, kernel_size=2,
                                               dilation=dilation, gain_nonlinearity="tanh")
        self.conditional_conv = Conv1dXavier(conv_channels, conv_channels,
                                             kernel_size=1, gain_nonlinearity="tanh")

        self.dilated_gate_conv = CausalConv1dXavier(conv_channels, conv_channels, kernel_size=2,
                                                    dilation=dilation, gain_nonlinearity="sigmoid")
        self.conditional_gate_conv = Conv1dXavier(conv_channels, conv_channels,
                                                  kernel_size=1, gain_nonlinearity="sigmoid")

        self.final_conv = Conv1dXavier(conv_channels, conv_channels, kernel_size=1, gain_nonlinearity="relu")

    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        """
        Perform convolutions and apply gates and return result for residual and skip connections
        :param x: tensor of shape (bs, n_channels, seq_len) from previous block
        :param condition: tensor of shape (bs, n_channels, seq_len) -- spectrogram to condition on
        :return: tensor of shape (bs, n_channels, seq_len)
        """
        x = torch.tanh(self.dilated_conv(x) + self.conditional_conv(condition))
        gate = torch.sigmoid(self.dilated_gate_conv(x) + self.conditional_gate_conv(condition))
        x = x * gate
        x = self.final_conv(x)
        return x


class WaveNet(pl.LightningModule):
    """
    WaveNet model from https://arxiv.org/pdf/1609.03499.pdf
    :param n_mu_law: number of channels in mu_law embedding
    :param n_mels: n_mels from Melspec transform
    :param n_fft: n_fft from Melspec transform
    :param hop_length: hop_length from Melspec transform
    :param n_layers: Number of layers in the network
    :param dilation_cycle: When dilation comes back to 1 (eg, for 5: 2, 4, 8, 16, 32, 2, 4, ...)
    :param conv_channels: Number of channels in convolutions
    :param optimizer_lr: Learning rate for Adam optimizer
    """
    def __init__(self, n_mu_law: int, n_mels: int, n_fft: int, hop_length: int, n_layers: int,
                 dilation_cycle: int, conv_channels: int, optimizer_lr: float) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.n_mu_law = n_mu_law
        self.optimizer_lr = optimizer_lr

        self.melspec_upsampler = nn.ConvTranspose1d(n_mels, conv_channels, kernel_size=n_fft + 1, padding=n_fft // 2,
                                                    stride=hop_length, output_padding=hop_length - 1)

        self.input_conv = Conv1dXavier(1, conv_channels, kernel_size=1)


        residual_blocks_list = []
        for k in range(n_layers):
            residual_blocks_list.append(ResidualBlock(conv_channels, dilation=2 ** (k % dilation_cycle)))
        self.residual_blocks = nn.ModuleList(residual_blocks_list)

        self.output_conv = nn.Sequential(
            nn.ReLU(),
            Conv1dXavier(conv_channels, conv_channels, kernel_size=1, gain_nonlinearity="relu"),
            nn.ReLU(),
            Conv1dXavier(conv_channels, n_mu_law, kernel_size=1),
        )

    def forward(self, spectrogram: torch.Tensor, waveform: torch.Tensor) -> torch.Tensor:
        """
        Calculates forward pass.
        :param spectrogram: tensor of shape (bs, n_fft, spec_len) -- spectrogram to condition on
        :param waveform: tensor of shape (bs, 1, seq_len) -- ground truth waveform
        :return: tensor of shape (bs, n_mu_law, seq_len) -- categorical distribution on ground truth waveform
        """
        bs, _, seq_len = waveform.shape
        waveform = waveform.float()
        waveform = torch.cat([torch.zeros([bs, 1, 1], device=self.device),
                              waveform[:, :, :-1]], dim=2)
        spectrogram = self.melspec_upsampler(spectrogram)[:, :, :waveform.shape[2]]
        waveform = self.input_conv(waveform)

        prev = waveform
        result = torch.zeros_like(spectrogram, device=self.device)  # Summed skip connections
        for block in self.residual_blocks:
            block_result = block(prev, spectrogram)
            result += block_result
            block_result += prev
            prev = block_result

        result = self.output_conv(result)

        return result
    
    def inference(self, spectrogram):
        """
        Calculates waveform from spectrogram
        :param spectrogram:
        :return: tensor of shape (bs, 1, seq_len) -- predicted waveform
        """
        spectrogram = self.melspec_upsampler(spectrogram)
        bs, _, seq_len = spectrogram.shape

        cur_waveform = torch.zeros([bs, 1, 1], device=self.device)
        while cur_waveform.shape[2] < spectrogram.shape[2]:
            print(f"{cur_waveform.shape[2]}/{spectrogram.shape[2]}")
            input_waveform = self.input_conv(cur_waveform)
            cur_result = torch.zeros_like(input_waveform, device=self.device)
            prev = input_waveform
            for block in self.residual_blocks:
                block_result = block(prev, spectrogram[:, :, :prev.shape[2]])
                cur_result += block_result
                block_result += prev
                prev = block_result
            cur_result = self.output_conv(cur_result)
            cur_result = cur_result[:, :, [-1]].argmax(dim=1, keepdim=True)
            cur_waveform = torch.cat([cur_waveform, cur_result], dim=2)

        return cur_waveform

    def step(self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
             batch_idx: int, inference: bool) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Pass batch to network, calculate losses and return total loss with gt and predicted spectrograms
        """
        waveform, spectrogram, _, _ = batch
        pred_waveform = self(spectrogram, waveform)

        loss = nn.CrossEntropyLoss()(pred_waveform, waveform.squeeze(1))

        return loss, waveform, pred_waveform, spectrogram

    def training_step(self, batch, batch_idx):
        loss, _, _, _, = self.step(batch, batch_idx, inference=False)

        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        # Calculate losses and results in training and inference modes
        val_loss, true_waveform, val_pred_waveform, spectrogram = self.step(batch, batch_idx, inference=False)
        val_pred_waveform = val_pred_waveform.argmax(dim=1)
        true_waveform = torchaudio.functional.mu_law_decoding(true_waveform, self.n_mu_law).squeeze(1)
        val_pred_waveform = torchaudio.functional.mu_law_decoding(val_pred_waveform, self.n_mu_law).squeeze(1)

        true_audio = [wandb.Audio(wav.detach().cpu(), sample_rate=22050) for wav in true_waveform]
        gen_audio = [wandb.Audio(wav.detach().cpu(), sample_rate=22050) for wav in val_pred_waveform]

        # Plot true and predicted on one image
        fig, axs = plt.subplots(len(true_audio))
        for ax, true, gen in zip(axs, true_waveform, val_pred_waveform):
            ax.plot(true.detach().cpu(), c="C0")
            ax.scatter(np.arange(len(gen)), gen.detach().cpu(), c="C1", s=0.01)

        self.logger.experiment.log({"True audio": true_audio,
                                    "Generated audio": gen_audio,
                                    "True and Gen compared": plt}, commit=False)
        self.log("val_loss", val_loss)

        return val_loss, true_waveform, val_pred_waveform, spectrogram

    # def validation_epoch_end(self, outputs):
    #     val_loss, true_waveform, val_pred_waveform, spectrogram = outputs[-1]
    #     inf_waveform = self.inference(spectrogram[:4])
    #     inf_waveform = torchaudio.functional.mu_law_decoding(inf_waveform, self.n_mu_law).squeeze(1)
    #
    #     inf_audio = [wandb.Audio(wav.detach().cpu(), sample_rate=22050) for wav in inf_waveform]
    #     self.logger.experiment.log({"Inferenced audio": inf_audio}, commit=False)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.optimizer_lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=5, min_lr=1e-5)
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "val_loss",
        }
