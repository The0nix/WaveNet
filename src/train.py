import hydra
import numpy as np
import pytorch_lightning as pl
import torch.utils.data as torchdata
from omegaconf import DictConfig, OmegaConf

import core


@hydra.main(config_path="../config", config_name="config")
def main(cfg: DictConfig):
    core.utils.fix_seeds(cfg.common.seed)

    # Define datasets and dataloaders:
    transforms = core.utils.get_transforms(cfg.train_transforms)
    inference_transforms = core.utils.get_transforms(cfg.inference_transforms)
    train_dataset = core.dataset.LJSPEECH(root=hydra.utils.to_absolute_path(cfg.data.root),
                                          n_mu_law=cfg.preprocessing.n_mu_law,
                                          crop_size=cfg.preprocessing.crop_size,
                                          transforms=transforms)
    val_dataset = core.dataset.LJSPEECH(root=hydra.utils.to_absolute_path(cfg.data.root),
                                        n_mu_law=cfg.preprocessing.n_mu_law,
                                        crop_size=cfg.preprocessing.crop_size,
                                        transforms=inference_transforms)
    dataset_lengths = np.array([len(train_dataset.get_transcript(idx)) for idx in range(len(train_dataset))])

    # Split data with stratification
    train_idx, val_idx = core.utils.get_split(train_dataset, train_size=cfg.data.train_size, random_state=cfg.common.seed)
    train_dataset = torchdata.Subset(train_dataset, train_idx)
    val_dataset = torchdata.Subset(val_dataset, val_idx)

    # Create sampler by transcription lengths
    train_dataset_lengths = dataset_lengths[train_idx]
    train_sampler = core.dataset.RandomBySequenceLengthSampler(train_dataset_lengths,
                                                               cfg.training.batch_size,
                                                               percentile=0.98)
    # Create dataloaders
    collate_fn = core.utils.PadCollator(np.log(cfg.preprocessing.clip_min_value), 0)
    train_dataloader = torchdata.DataLoader(train_dataset,
                                            batch_sampler=train_sampler,
                                            num_workers=cfg.training.num_workers,
                                            collate_fn=collate_fn)
    val_dataloader = torchdata.DataLoader(val_dataset,
                                          batch_size=cfg.training.batch_size,
                                          num_workers=cfg.training.num_workers,
                                          collate_fn=collate_fn, shuffle=False)

    # Define model
    if "checkpoint_path" in cfg.model:
        model = core.model.WaveNet.load_from_checkpoint(hydra.utils.to_absolute_path(cfg.model.checkpoint_path))
    else:
        model = core.model.WaveNet(n_mu_law=cfg.preprocessing.n_mu_law,
                                   n_mels=cfg.preprocessing.n_mels,
                                   n_fft=cfg.preprocessing.n_fft,
                                   hop_length=cfg.preprocessing.hop_length,
                                   n_layers=cfg.model.n_layers,
                                   dilation_cycle=cfg.model.dilation_cycle,
                                   conv_channels=cfg.model.conv_channels,
                                   optimizer_lr=cfg.optimizer.lr)

    # Define logger and trainer
    wandb_logger = pl.loggers.WandbLogger(project=cfg.wandb.project)
    wandb_logger.watch(model, log="gradients", log_freq=cfg.wandb.log_freq)
    checkpoint_callback = pl.callbacks.ModelCheckpoint(monitor="val_loss", save_last=True, save_top_k=1)
    trainer = pl.Trainer(max_epochs=cfg.training.n_epochs, gpus=cfg.training.gpus,
                         logger=wandb_logger, default_root_dir="checkpoints",
                         checkpoint_callback=checkpoint_callback,
                         val_check_interval=cfg.training.val_check_interval)

    # FIT IT!
    trainer.fit(model, train_dataloader, val_dataloader)


if __name__ == "__main__":
    main()
