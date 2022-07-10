class PathsConfig:
    audios = '/root/audios'
    alignments = '/root/alignments'
    noises = '/root/noises'
    rirs = '/root/rirs'
    augmented = '/root/augmented_audios'
    features_labels = '/root/features_labels'
    infer = '/root/infer'
    meta = 'meta/'

target_sample_rate = 16000
hop_size = 0.01 #sec
webrtc_far = 0.0619 #for comparison with webrtc 

class AugmentConfig:
    noise_prob = 0.6
    snr_range = [0, 20]
    noise_min_length = 8 #sec
    reverb_prob = 0.6
    
class TrainConfig:
    device = 'cpu'
    n_epochs = 200
    n_steps_per_epoch = 1000
    lr = 3e-6

class TypesConfig:
    audios = 'flac'
    noises = 'wav'
    rirs = 'wav'
    infer = 'wav'