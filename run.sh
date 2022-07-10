#!/bin/bash

docker run \
-v $(pwd):/root/workdir \
-v /home/eugene/Datasets/LibriSpeech:/root/audios \
-v /home/eugene/Datasets/LibriSpeech_alignments:/root/alignments \
-v /home/eugene/Datasets/MUSAN/noise:/root/noises \
-v /home/eugene/Datasets/sim_rir_16k:/root/rirs \
-v /home/eugene/Datasets/LibriSpeech_augmented/audios:/root/augmented_audios \
-v /home/eugene/Datasets/LibriSpeech_augmented/features_labels:/root/features_labels \
-v /home/eugene/Datasets/for_devs:/root/infer \
-it vkvad
