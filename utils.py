from pathlib import Path
from scipy.optimize import brentq
from sklearn.metrics import roc_curve
from scipy.interpolate import interp1d
import torch
from torch.utils.data import Dataset
import torchaudio
import os
import json
from typing import List

from config import PathsConfig, target_sample_rate, webrtc_far


class FeaturesLabelsDataset(Dataset):
    '''
    Loads calculated features and labels from PathsConfig.features_labels using info from meta json.
    '''
    def __init__(self, mode: str = 'train'):
        '''
        Parameters:
            mode (str): train or val
        '''
        super().__init__()
        self.meta = json.load(open(meta_path(mode), 'r'))
        
    def __len__(self):
        return len(self.meta)

    def __getitem__(self, i: int):
        features = torch.load(self.meta[i]['feature_path'])
        labels = torch.load(self.meta[i]['label_path'])
        if features.shape[0] - labels.shape[0] == 1:
            labels = torch.vstack([labels, torch.zeros((1, 1))])
        return features, labels


def metrics(all_labels: List[float], all_outputs: List[float]) -> dict:
    '''
    Calculates the metrics using predicted scores and corresponding true labels.
    Parameters:
        all_labels (list[float]): true labels, possible values are 0 and 1
        all_outputs (list[float]): pridicted scores in 0--1 range
    Returns:
        results (dict): dictionary with metrics:
            eer: EER (FAR = FRR)
            eer_thr: threshold at which EER is reached
            frfa1: FRR when FAR is 1%
            frfa1_thr: threshold at which FAR is 1%
            fafr1: FAR when FRR is 1%
            fafr1_thr: threshold at which FRR is 1%
    '''
    fpr, tpr, thresholds = roc_curve(all_labels, all_outputs)
    frfa_curve = interp1d(fpr, 1 - tpr)
    fafr_curve = interp1d(1 - tpr, fpr)
    thr_curve = interp1d(fpr, thresholds)
    results = dict()
    results['eer'] = brentq(lambda x: frfa_curve(x) - x, 0., 1.)
    results['eer_thr'] = thr_curve(results['eer'])
    results['frfa1'] = frfa_curve(0.01)
    results['frfa1_thr'] = thr_curve(0.01)
    results['fafr1'] = fafr_curve(0.01)
    results['fafr1_thr'] = thr_curve(results['fafr1'])
    results['frfa_webrtc'] = frfa_curve(webrtc_far)
    results['frfa_webrtc_thr'] = thr_curve(webrtc_far)
    results['fars'] = fpr
    results['frrs'] = 1 - tpr
    results['thrs'] = thresholds
    return results


def meta_path(mode: str = 'train') -> Path:
    '''
    Returns the meta json path for a given mode.
    Parameters:
        mode (str): train or val
    Returns:
        json_path (Path): json location
    '''
    os.makedirs(PathsConfig.meta, exist_ok=True)
    return (Path(PathsConfig.meta) / mode).with_suffix('.json')


def read_audio(audiopath: str) -> torch.Tensor:
    '''
    Reads the audio, checks its sample rate and returns a tensor with samples.
    Parameters:
        audiopath (str): path to the audio file to read
    Returns:
        samples (torch.Tensor): tensor of loaded audio samples
    '''
    samples, sample_rate = torchaudio.load(audiopath)
    assert sample_rate == target_sample_rate
    return samples
