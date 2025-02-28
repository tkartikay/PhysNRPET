import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from PIL import Image
from pytorch_lightning.callbacks import LearningRateMonitor
import os
from pytorch_lightning.loggers import WandbLogger
import wandb
from Net import SirenPNR
from PetDatasets import DynPETQSDataset, Val2DPETDataset
torch.autograd.set_detect_anomaly(True)
torch.set_float32_matmul_precision('medium')
torch.manual_seed(seed=0)
torch.cuda.manual_seed(seed=0)
torch.mps.manual_seed(seed=0)

import multiprocessing as mp
mp.set_start_method('spawn', force=True)

def main_inr():
    device = torch.device("mps")#"cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    train_data = DynPETQSDataset(sample_size=128*128)
    print("len train: ", len(train_data))
    val_data = Val2DPETDataset()
    train_loader = DataLoader(train_data, batch_size=128, pin_memory=True, num_workers=6, persistent_workers=True)
    val_loader = DataLoader(val_data, batch_size=len(val_data), pin_memory=True, num_workers=4, persistent_workers=True)
    wandb_logger = WandbLogger(project="PhysNRPET")
    lr_monitor = LearningRateMonitor(logging_interval='epoch')

    mapping_size = 256
    B_gauss = torch.randn((mapping_size, 3)).to(device) #coords
    B_gausshu = torch.randn((mapping_size, 4)).to(device) #coords + hu
    model =SirenPNR(in_features=mapping_size*2, B=B_gauss*10, out_features=4,
                hidden_layers=3)
    # model =SirenPNR(in_features=mapping_size*2, B=B_gausshu*10, out_features=4,
    #             hidden_layers=3)
    # model =SirenPNR(in_features=mapping_size*2+4096, B=B_gauss*10, out_features=4,
    #             hidden_layers=3)

    trainer = pl.Trainer(max_epochs=101,
                        logger=wandb_logger,
                        callbacks=[lr_monitor],
                        check_val_every_n_epoch=10,
                        gradient_clip_val=100,
                        )
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    torch.save(model, 'model.pt')

if __name__ == "__main__":
    main_inr()