import torch
from torch import nn
import numpy as np
import torchaudio
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
import os
import sys
import time
from tqdm import tqdm
from speechbrain.lobes.features import Fbank


from utils import metrics, read_audio, FeaturesLabelsDataset
from config import hop_size, webrtc_far, TrainConfig


device = TrainConfig.device


class VAD(nn.Module):
    '''
    Implements chosen architecture of VAD and training/evaluation methods.
    '''
    def __init__(self, weights_path: str = None, hidden_dim: int = 512, context_frames: int = 5):
        '''
        Parameters:
            weights_path (str): location of the trained state dict to load, weights are initialised randomly if None
            hidden_dim (int): number of weights on hidden layers
            context_frames (int): number of close frames to extract features from for the model to use context information.
        '''
        super().__init__()
        self.feature_extractor = Fbank(hop_length=hop_size*1000)
        self.model = nn.Sequential(
            nn.Linear((context_frames * 2 + 1) * 40, hidden_dim), 
            nn.BatchNorm1d(hidden_dim), 
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), 
            nn.BatchNorm1d(hidden_dim), 
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )
        if weights_path:
            self.model.load_state_dict(torch.load(weights_path))
        self.loss = nn.BCELoss()
        self.context_frames = context_frames
        self.to(device)
        
    def add_context(self, features: torch.Tensor) -> torch.Tensor:
        '''
        Concatenates features from close frames to the current frame to use context information.
        Parameters:
            features (torch.Tensor): features without context
        Returns:
            features_context (torch.Tensor): features with context
        '''
        n = features.shape[0]
        padded = torch.vstack([
            torch.zeros((self.context_frames, features.shape[1])), 
            features, 
            torch.zeros((self.context_frames, features.shape[1]))
        ])
        features_context = torch.hstack([padded[i:n+i] for i in range(self.context_frames * 2 + 1)])
        return features_context
        
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        '''
        Predicts VAD scores for the input features (without context).
        Parameters:
            features (torch.Tensor): features without context
        Returns:
            output (torch.Tensor): VAD scores for each feature frame
        '''
        context = self.add_context(features)
        return self.model(context)
    
    def fit(self, 
            train_data: FeaturesLabelsDataset, 
            val_data: FeaturesLabelsDataset, 
            num_epochs: int = 10, 
            steps_per_epoch: int = 500, 
            lr: float = 1e-5):
        '''
        Trains the model on train_data and calculates metrics on val_data each epoch.
        Parameters:
            train_data (FeaturesLabelsDataset): training dataset
            val_data (FeaturesLabelsDataset): validation dataset
            num_epochs (int): number of epochs to train the model
            steps_per_epoch (int): number of audios per epoch to train on
            lr (float): learning rate
        '''
        train_dl = DataLoader(train_data, shuffle=True)
        opt = Adam(self.model.parameters(), lr=lr)
        writer = SummaryWriter()
        logdir = writer.log_dir
        step = 0
        for epoch in range(num_epochs):
            print(f'Epoch: {epoch}')
            self.model.train()
            step_loss = []
            for features, labels in tqdm(train_dl, file=sys.stdout, desc='Training', total=steps_per_epoch):
                features = features.squeeze(0)
                labels = labels.squeeze(0)
                opt.zero_grad()
                outputs = self.forward(features.to(device))
                loss = self.loss(outputs, labels.to(device))
                loss.backward()
                opt.step()
                writer.add_scalar('loss/trainbatch', loss.item(), step)
                step_loss.append(loss.item())
                step += 1
                if step % steps_per_epoch == 0:
                    break
            epoch_train_loss = np.mean(step_loss)
            writer.add_scalars('loss/epoch', {'train': epoch_train_loss}, epoch)
            print(f'Train loss: {epoch_train_loss}')
            
            val = self.validate(val_data)
            writer.add_scalars('loss/epoch', {'val': val['loss']}, epoch)
            writer.add_scalar('metrics/eer', val['eer'], epoch)
            writer.add_scalar('metrics/fafr1', val['fafr1'], epoch)
            writer.add_scalar('metrics/frfa1', val['frfa1'], epoch)
            writer.add_scalar(f'metrics/frfa{webrtc_far*100}', val['frfa_webrtc'], epoch)
            writer.add_scalar('thresholds/eer', val['eer_thr'], epoch)
            writer.add_scalar('thresholds/fafr1', val['fafr1_thr'], epoch)
            writer.add_scalar('thresholds/frfa1', val['frfa1_thr'], epoch)
            writer.add_scalar(f'thresholds/frfa{webrtc_far*100}', val['frfa_webrtc_thr'], epoch)
            print(f"Val loss: {val['loss']}")
            print(f"Val EER: {val['eer']} (thr = {val['eer_thr']})")
            print(f"Val FA(FR=1%): {val['fafr1']} (thr = {val['fafr1_thr']})")
            print(f"Val FR(FA=1%): {val['frfa1']} (thr = {val['frfa1_thr']})")
            print()
            
            os.makedirs(f'{logdir}/models', exist_ok=True)
            torch.save(self.model.state_dict(), f'{logdir}/models/{epoch}.pt')
        writer.close()
    
    @torch.no_grad()
    def validate(self, val_data: FeaturesLabelsDataset) -> dict:
        '''
        Validates the model on val_data and returns metrics.
        Parameters:
            val_data (FeaturesLabelsDataset): validation dataset
        Returns:
            results (dict): dict with metrics, see utils.metrics for more info
        '''
        self.model.eval()
        val_dl = DataLoader(val_data, shuffle=False)
        step_loss = []
        all_labels = []
        all_outputs = []
        for features, labels in tqdm(val_dl, file=sys.stdout, desc='Validation'):
            features = features.squeeze(0)
            labels = labels.squeeze(0)
            outputs = self.forward(features.to(device))
            loss = self.loss(outputs, labels.to(device))
            step_loss.append(loss.item())
            all_labels.extend(labels.tolist())
            all_outputs.extend(outputs.tolist())
        results = metrics(all_labels, all_outputs)
        results['loss'] = np.mean(step_loss)
        return results
        
    @torch.no_grad()
    def __call__(self, path: str, threshold: float = None) -> np.ndarray:
        '''
        Predict VAD scores for input audio file.
        Parameters:
            path (str): path to the input audio file
            threshold (float): optional threshold to make the output binary
        Returns:
            output (np.ndarray): array of predictions (bools if threshold is given, float probabilities else)
        '''
        self.model.eval()
        samples = read_audio(path)
        features = self.feature_extractor(samples).squeeze(0)
        output = self.forward(features).squeeze(1).numpy()
        if threshold is not None:
            return output > threshold
        else:
            return output
        
        
