from typing import Any
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import pytorch_lightning as pl
import wandb
from Utils import torch_interp_1d, TAC_2TC_KM

device = torch.device("mps")#"cuda" if torch.cuda.is_available() else "cpu")

class SineLayer(nn.Module):
    # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of omega_0.

    # If is_first=True, omega_0 is a frequency factor which simply multiplies the activations before the
    # nonlinearity. Different signals may require different omega_0 in the first layer - this is a
    # hyperparameter.

    # If is_first=False, then the weights will be divided by omega_0 so as to keep the magnitude of
    # activations constant, but boost gradients to the weight matrix (see supplement Sec. 1.5)

    def __init__(self, in_features, out_features, bias=True,
                 is_first=False, omega_0=30):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first

        # self.bn = nn.BatchNorm1d(num_features=out_features)

        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)

        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features,
                                             1 / self.in_features)
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0,
                                             np.sqrt(6 / self.in_features) / self.omega_0)

    def forward(self, input):
        return torch.sin(self.omega_0 * self.linear(input))

    def forward_with_intermediate(self, input):
        # For visualization of activation distributions
        intermediate = self.omega_0 * self.linear(input)
        return torch.sin(intermediate), intermediate
    
class SirenPNR(pl.LightningModule):
    def __init__(self, in_features=3, hidden_features=512, hidden_layers=7, out_features=1,
                 B=None, outermost_linear=False,
                 first_omega_0=30, hidden_omega_0=30.):
        super().__init__()

        self.B = B
        self.net = []
        self.net.append(SineLayer(in_features, hidden_features,
                                  is_first=True, omega_0=first_omega_0))
        for i in range(hidden_layers):
            self.net.append(SineLayer(hidden_features, hidden_features,
                                      is_first=False, omega_0=hidden_omega_0))
        self.net = nn.Sequential(*self.net)
        self.k_linear = nn.Linear(hidden_features, out_features)
        with torch.no_grad():
            self.k_linear.weight.uniform_(-np.sqrt(6 / hidden_features) / hidden_omega_0,
                                           np.sqrt(6 / hidden_features) / hidden_omega_0)
        idif_path = 'IDIF.txt'
        sample_time = np.loadtxt(idif_path, delimiter="\t", usecols=[0], skiprows=1)
        self.sample_time = torch.tensor(sample_time, dtype=torch.float32).to(device)
        # print("sample time: ", self.sample_time)
        idif = np.loadtxt(idif_path, delimiter="\t", usecols=[1], skiprows=1)
        idif_tensor = torch.tensor(idif, dtype=torch.float32).to(device)
        print("max idif: ", idif_tensor.max()) # =278 kBq/ml
        idif = idif_tensor / 278
        step = 0.03
        self.t = torch.Tensor(torch.arange(self.sample_time[0], self.sample_time[-1], step)).to(device)
        self.idif_interp = torch_interp_1d(self.t, self.sample_time, idif).to(device)
        self.matching_indices = []
        for st in self.sample_time:
            idx = torch.argmin(torch.abs(self.t - st))
            self.matching_indices.append(idx)
        self.matching_indices = torch.stack(self.matching_indices)

    def forward(self, x):#xfm when using foundation model features 
        # x = xfm[:,0:3]
        # fm = xfm[:,3:]
        if self.B is not None:
            x = torch.matmul(2. * torch.pi * x, self.B.T).to(device)
            x = torch.cat([torch.sin(x), torch.cos(x)], -1).to(device)
        
        xf = self.net(x)
        ki = torch.nn.Softplus(beta=5)(self.k_linear(xf))
        return ki
    
    def configure_optimizers(self):
        optim = torch.optim.Adam(self.parameters(), lr=1e-5)
        return optim
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        if self.current_epoch == 0 and batch_idx == 0:
            print("Batch 0 inputs: ", x.shape, y.shape)

        k_hat = self(x)

        vb_loss = torch.nn.functional.relu(k_hat[:,3] - 1.0).mean() # enforce Vb between 0 and 1

        idif_interp = self.idif_interp.repeat(k_hat.shape[0], 1)
        C_km_est = TAC_2TC_KM(idif_interp, self.t, k_hat, step=0.03)
        # C_km_est = torch.nan_to_num(C_km_est, nan=0.0, posinf=10, neginf=-1e3)
        C_km = C_km_est[:, self.matching_indices]
        km_loss = nn.MSELoss()(C_km, y.squeeze(2)) # kinetic model loss at measured TAC time points
        self.log('train_loss', km_loss + vb_loss, on_epoch=True)
        return km_loss + vb_loss
    
    def validation_step(self, batch):
        x, y = batch
        k_hat = self(x)
        h, w = 170, 170
        ki = k_hat.view(h,w,4)
        # pass
        return

