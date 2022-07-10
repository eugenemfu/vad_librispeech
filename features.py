import os
import argparse
from pathlib import Path
from tqdm import tqdm
import torch
import torchaudio
from speechbrain.lobes.features import Fbank
import json
from typing import List

from config import PathsConfig, hop_size
from utils import read_audio, meta_path


def main(modes: List[str]):
    '''
    Calculates features of audios listed in meta jsons and saves them to PathsConfig.features_labels.
    Adds saved feature paths to the meta jsons.
    Parameters:
        modes (list[str]): ["train"] or ["val"] or ["train", "val"]
    '''
    feature_extractor = Fbank(hop_length=hop_size*1000)
    
    for mode in modes:
        meta = json.load(open(meta_path(mode), 'r'))
    
        for i in tqdm(range(len(meta)), desc=mode):
            featurepath = meta[i]['label_path'].replace('.labels.pt', '.features.pt')
            meta[i]['feature_path'] = featurepath

            samples = read_audio(meta[i]['audio_path'])
            features = feature_extractor(samples).squeeze(0)
            torch.save(features, featurepath)
            
    json.dump(meta, open(meta_path(mode), 'w'))
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--modes', nargs='+', type=str, required=True,
                        help='List of modes to extract features (train, val)')
    args = parser.parse_args()
    main(args.modes)