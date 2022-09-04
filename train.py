import torch
import torchvision
from torch.utils.data import DataLoader
import torch.functional as F
import torchvision.transforms as transforms
from EMP_data import EmpDataset


#hyper-parameters
BATCH_SIZE = 20
LEARNING_RATE =1e-4
NUM_EPOCHS = 3
NUM_WORKERS =1
PIN_MEMORY= False





train_dataloder = DataLoader(dataset=EmpDataset,batch_size=BATCH_SIZE,shuffle=True)

