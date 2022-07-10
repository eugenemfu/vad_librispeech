from collections import defaultdict
import torch
import torchaudio
from pathlib import Path
import numpy as np
import os
import json
import argparse
from tqdm import tqdm
import textgrid
from typing import List

from config import PathsConfig, AugmentConfig, TypesConfig, hop_size, target_sample_rate
from utils import read_audio, meta_path


class Augmentor:
    '''
    Reads noises and RIRs from the datasets (in PathsConfig) and applies them in methods add_noise and add_reverb.
    '''
    def __init__(self):
        self.noise_set = []
        for dirpath, dirnames, filenames in os.walk(PathsConfig.noises):
            for f in filenames:
                if f.endswith(f".{TypesConfig.noises}"):
                    path = Path(dirpath) / f
                    metadata = torchaudio.info(path)
                    assert metadata.sample_rate == target_sample_rate
                    if metadata.num_frames / target_sample_rate > AugmentConfig.noise_min_length:
                        self.noise_set.append(path)
        
        self.rir_set = []
        for dirpath, dirnames, filenames in os.walk(PathsConfig.rirs):
            self.rir_set.extend([Path(dirpath) / f for f in filenames if f.endswith(f".{TypesConfig.rirs}")])

    def add_reverb(self, samples: torch.Tensor) -> torch.Tensor:
        '''
        Applies random reverberation from the RIR dataset to the input samples.
        '''
        rir_path = np.random.choice(self.rir_set)
        rir = read_audio(rir_path)
        if rir.shape[1] >= samples.shape[1]:
            rir = rir[:, :samples.shape[1]]
        else:
            rir = torch.hstack([rir, torch.zeros((1, samples.shape[1] - rir.shape[1]))])
        
        signal_fft = torch.fft.rfft(samples)
        rir_fft = torch.fft.rfft(rir)
        reverbed = torch.fft.irfft(signal_fft * rir_fft)
        
        power_before = torch.linalg.vector_norm(samples).item() / samples.shape[1]
        power_after = torch.linalg.vector_norm(reverbed).item() / reverbed.shape[1]
        return reverbed * power_before / power_after

    def add_noise(self, samples: torch.Tensor) -> torch.Tensor:
        '''
        Mixes random noise from the noise dataset to the input samples.
        '''
        noise_path = np.random.choice(self.noise_set)
        noise = read_audio(noise_path)
        if noise.shape[1] >= samples.shape[1]:
            start = np.random.randint(0, noise.shape[1] - samples.shape[1])
            noise = noise[:, start:start+samples.shape[1]]
        else:
            mult = samples.shape[1] // noise.shape[1]
            rest = samples.shape[1] % noise.shape[1]
            noise = torch.hstack([noise.repeat(1, mult), noise[:, :rest]]) 
        
        signal_power = (samples @ samples.T).item() / samples.shape[1]
        noise_power = (noise @ noise.T).item() / noise.shape[1]

        snr = np.random.sample() * (AugmentConfig.snr_range[1] - AugmentConfig.snr_range[0]) + AugmentConfig.snr_range[0]
        factor = np.sqrt(10 ** (- snr / 10) * signal_power / noise_power)
        return samples + noise * factor
    

def get_labels_from_textgrid(textgrid_path: str, len_samples: int) -> torch.Tensor:
    '''
    Reads a textgrid file with word alignments and transforms it into binary VAD labels.
    Parameters:
        textgrid_path (str): path to the alignment textgrid file
        len_samples (int): length of the audio (in frames)
    Returns:
        labels (torch.Tensor): binary VAD labels for each hop_size sec window
    '''
    labels = torch.zeros(int(len_samples / target_sample_rate / hop_size) + 1)
    try:
        for interval in textgrid.TextGrid.fromFile(textgrid_path)[0]:
            if interval.mark != '':
                start = int(round(interval.minTime / hop_size))
                end = int(round(interval.maxTime / hop_size))
                labels[start:end] = 1
    except FileNotFoundError:
        return None
    return labels.unsqueeze(1)


def main(mode: str, list_paths: List[str]):
    '''
    Reads lists of audios used for training/validation and:
    1) augments them with noises and RIRs using Augmentor and saves new audios to the PathsConfig.augmented,
    2) gets VAD labels from alignments using get_labels_from_textgrid function and saves them to PathsConfig.features_labels,
    3) creates json dataset description in PathsConfig.meta.
    
    Parameters:
        mode (str): "train" or "val"
        list_paths (list[str]): paths to the lists of considered audios
    '''
    meta = []
    augmentor = Augmentor()
    
    for list_path in list_paths:
        list_name = Path(list_path).stem
        with open(list_path, 'r') as f:
            list_path = f.read().splitlines()
        
        for path in tqdm(list_path, desc=list_name):
            audiopath = Path(PathsConfig.audios) / path
            alignpath = (Path(PathsConfig.alignments) / path).with_suffix('.TextGrid')
            
            samples = read_audio(audiopath)
            labels = get_labels_from_textgrid(alignpath, samples.shape[1])
            if labels is None:
                print('An alignment is missing! Skipping this audio.')
                continue
            
            added_reverb = False
            if np.random.sample() < AugmentConfig.reverb_prob:
                added_reverb = True
                samples = augmentor.add_reverb(samples)
            
            added_noise = False
            if np.random.sample() < AugmentConfig.noise_prob:
                added_noise = True
                samples = augmentor.add_noise(samples)

            outputpath = (Path(PathsConfig.augmented) / path).with_suffix(f'.{TypesConfig.audios}')
            labelpath = (Path(PathsConfig.features_labels) / path).with_suffix(f'.labels.pt')
            
            meta.append(dict(
                audio_path=str(outputpath),
                label_path=str(labelpath),
                origin=str(audiopath),
                added_noise=added_noise,
                added_reverb=added_reverb,
            ))
            
            os.makedirs(labelpath.parent, exist_ok=True)
            os.makedirs(outputpath.parent, exist_ok=True)
            torch.save(labels, labelpath)
            torchaudio.save(outputpath, samples, target_sample_rate, format=TypesConfig.audios)
    
    json.dump(meta, open(meta_path(mode), 'w'))
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--mode', type=str, default='train',
                        help='train or val')
    parser.add_argument('-l', '--lists', nargs='+', type=str, required=True,
                        help='Lists of the consedered files')
    args = parser.parse_args()
    assert args.mode in ['train', 'val']
    for path in args.lists:
        assert os.path.isfile(path)
    main(args.mode, args.lists)