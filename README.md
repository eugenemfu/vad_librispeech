# VAD model trained on LibriSpeech

## Inference steps

To infer on single audio and visualise results, you can use Colab demo: 

...

To infer on folder of audios:

1. Build a docker container: ``./build.sh``

2. Be sure to mount desired folder to ``/root/infer`` by editing ``run.sh`` and changing the ``TypesConfig.infer`` audio type in ``config.py`` if needed.

3. Start docker: ``./run.sh``

4. Run ``python infer.py --model ...``

5. The output file with predicted VAD scores is ``meta/pred.txt``

## Training steps

1. Build a docker container: ``./build.sh``

2. Mount all needed data folders by editing ``run.sh``. These are:

    ``/root/audios`` - initial LibriSpeech corpus (https://www.openslr.org/12/)
    
    ``/root/alignments`` - corpus of word alignments for LibriSpeech (https://zenodo.org/record/2619474#.YsrzI0hBzJG)
    
    ``/root/noises`` and ``/root/rirs`` - any corpuses of noises and RIRs
    
    ``/root/augmented_audios`` - destination for saving augmentations
    
    ``/root/features_labels`` - destination for saving features and labels for training.

3. Review ``config.py`` and edit values if needed.

4. Start docker: ``./run.sh``

5. Create files with lists of audios which are going to be used in training/validation. Each line must be a path to an audio file from the dataset root (``/root/audios``). You can use these commands:

```
cd /root/audios
for f in dev-clean/*/*/*.flac; do echo $f >> /root/workdir/lists/dev-clean.txt; done
for f in train-clean-100/*/*/*.flac; do echo $f >> /root/workdir/lists/train-clean-100.txt; done
for f in train-clean-360/*/*/*.flac; do echo $f >> /root/workdir/lists/train-clean-360.txt; done
cd /root/workdir
```

6. Augment files from created lists and create train/val protocols:

    ``python augment.py -m train -l lists/train-clean-*``
    
    ``python augment.py -m val -l lists/dev-clean.txt``
    
    This script uses random files from ``/root/noises`` and ``/root/rirs`` and saves obtained augmented audios to ``/root/augmented_audios``. Also this script reads ``/root/alignments`` and creates binary VAD labels that are saved to ``/root/features_labels``. JSON protocols used in training are saved in ``meta`` folder.
    
7. Extract features from augmented audios:

    ``python features.py -m train val``
    
    This script reads JSON files from ``meta`` folder, extracts features from listed audios, saves the features to ``/root/features_labels`` and adds info about features to JSON.
    
8. Train the model: 

    ``python train.py``
    
    You can use the stdout or TensorBoard to watch validation metrics. TensorBoard logs and trained models are saved to ``runs`` folder.
    
## Comparison with WebRTC

To compare the metrics with WebRTC, use ``webrtc.ipynb`` (before the last step of training) under any python<=3.7 environment with webrtcvad installed. Edit paths and run all cells. Take any FAR from the results and put into ``webrtc_far`` in ``config.py``. The training TensorBoard logs will contain FRR values for this FAR.