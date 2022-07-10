import numpy as np
import torch

from utils import FeaturesLabelsDataset
from model import VAD
from config import TrainConfig


if __name__ == '__main__':
    
    torch.manual_seed(0)
    np.random.seed(0)

    train_data = FeaturesLabelsDataset('train')
    val_data = FeaturesLabelsDataset('val')

    vad = VAD()
    vad.fit(train_data, 
            val_data, 
            num_epochs=TrainConfig.n_epochs, 
            steps_per_epoch=TrainConfig.n_steps_per_epoch, 
            lr=TrainConfig.lr)