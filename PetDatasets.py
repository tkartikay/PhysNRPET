import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset
import monai
from monai.transforms import LoadImaged, ScaleIntensityRanged, ToTensord
from natsort import natsorted

device = torch.device("mps")#"cuda" if torch.cuda.is_available() else "cpu")

class DynPETQSDataset(Dataset):
    def __init__(self, sample_size=2):
        seed = 0
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        np.random.seed(seed)

        self.idx = 0
        """
        Args:
            image_tensor (torch.Tensor): A 4D tensor representing the dynamic PET image volume.
        """

        data_dir = "/home/DynamicFDGPET/NIFTY"
        image_tensor = natsorted(glob.glob(os.path.join(data_dir, "PET_*")))
        image_dict = [
            {"image": image_name} for image_name in image_tensor
        ]

        transforms = monai.transforms.Compose([
            LoadImaged(keys=["image"]),
            ScaleIntensityRanged(keys=["image"], a_min=0, a_max=200000, b_min=0.0, b_max=1.0, clip=True),
            ToTensord(keys=["image"])
        ])

        dynpetdata = transforms(image_dict)
        image_tensor = torch.stack([entry['image'] for entry in dynpetdata], dim=0)
        # image_tensor = image_tensor[:, 94:350, 230:231, 204:460]
        # torch.save(image_tensor, 'liverslice09.pt')
        # image_tensor = torch.load('liverslice09.pt', weights_only=False)
        # ctimage_tensor = torch.load('ctliverslice09.pt', weights_only=False)
        # ctfm_tensor = torch.load('ctfmslice09.pt', weights_only=False).to("cpu")
        # Generate coordinates
        T, D, H, W = image_tensor.shape
        print("Image Tensor Shape: ", image_tensor.shape)
        print("N images : ", T)
        self.coords = torch.stack(torch.meshgrid(torch.arange(D),
                                  torch.arange(H), torch.arange(W)),
                                  -1).reshape(-1, 3).float()
        # Normalize coordinates to [-1, 1]
        self.coords = self.coords / torch.tensor([D/2, H/2, W/2]) - 1
        # Flatten the image volume to match coordinates
        self.intensities = image_tensor
        # self.huvalues = ctimage_tensor
        # self.fmvalues = ctfm_tensor.unsqueeze(1) / 10.0 # scale between 0 and 1
        self.sample_size = sample_size

    def __len__(self):
        return self.sample_size

    def __getitem__(self, _):
        idx = torch.randint(0, len(self.coords), (1,)).item() # uniform sampling

        # ctintensity = self.huvalues.reshape(-1,1)[idx]
        # hucoords = torch.cat((self.coords[idx], torch.tensor([ctintensity])), 0)

        # tfeatures = self.fmvalues.reshape(-1,self.fmvalues.shape[3])[idx, :]
        # fmcoords = torch.cat((self.coords[idx], ctfeatures), 0)
        
        # Fetch the intensity values at the selected flattened coordinate idx across all time points
        intensities = self.intensities.reshape(self.intensities.shape[0], -1)[:, idx].reshape(-1, 1)
        return self.coords[idx], intensities
        # return hucoords, intensities
        # return fmcoords, intensities

class Val2DPETDataset(Dataset):
    def __init__(self):
        torch.manual_seed(0)
        np.random.seed(seed=0)
        image = torch.load('liverslice09.pt', weights_only=False)
        img = image[61] #pick a frame to validate
        # Generate coordinates
        D, H, W = img.shape
        T = 1
        self.coords = torch.stack(torch.meshgrid(torch.arange(D),
                                  torch.arange(H), torch.arange(W)),
                                  -1)
        # Normalize coordinates to [-1, 1]
        self.coords = self.coords[:,0,:].unsqueeze(2).reshape(-1, 3).float() #60
        self.coords = self.coords / torch.tensor([D/2, H/2, W/2]) - 1
        # Flatten the image volume to match coordinates
        self.intensities = img[:,0,:].unsqueeze(2) #60
        del img
        # ctimage_tensor = torch.load('ctliverslice09.pt', weights_only=False)
        # self.huvalues = ctimage_tensor[:,0,:]
        # fmct_tensor = torch.load('ctfmslice09.pt', weights_only=False).to("cpu")
        # fmct_tensor = fmct_tensor.unsqueeze(1)
        # self.fmvalues = fmct_tensor[:,0,:,:] / 10.0

    def __len__(self):
        return len(self.coords)
    
    def __getitem__(self, index):
        intensity = self.intensities.reshape(-1, 1)[index]
        return self.coords[index], torch.tensor([intensity])

        # ctintensity = self.huvalues.reshape(-1,1)[index]
        # hucoords = torch.cat((self.coords[index], torch.tensor([ctintensity])), 0)
        # return hucoords, torch.tensor([intensity])
    
        # ctfeatures = self.fmvalues.reshape(-1,self.fmvalues.shape[2])[index,:]
        # fmcoords = torch.cat((self.coords[index], ctfeatures), 0)
        # return fmcoords, torch.tensor([intensity])
