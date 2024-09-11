import os
import torchaudio
import torch
import torch.nn.functional as F
import pandas as pd

def extract_log_mel_spectrogram(audio_path, sample_rate=16000, n_mels=40, target_length=51):
    waveform, sr = torchaudio.load(audio_path)

    # calculate Mel Spectrogram
    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=1024,
        win_length=640,
        hop_length=320,
        n_mels=n_mels
    )(waveform)

    # convert to log_mel
    log_mel_spectrogram = torch.log(mel_spectrogram + 1e-9)

    # cut and fill
    if log_mel_spectrogram.size(2) > target_length:
        log_mel_spectrogram = log_mel_spectrogram[:, :, :target_length]
    else:
        pad_amount = target_length - log_mel_spectrogram.size(2)
        log_mel_spectrogram = F.pad(log_mel_spectrogram, (0, pad_amount))

    return log_mel_spectrogram

def process_audio_files(audio_dirs, labels, output_csv, sample_rate=16000, n_mels=40, target_length=51):
    data = []
    file_index = 1  # Start numbering files from 1

    # creat path
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)

    for audio_dir, label in zip(audio_dirs, labels):
        for root, _, files in os.walk(audio_dir):
            for file in files:
                if file.endswith('.wav'):
                    file_path = os.path.join(root, file)

                    # calculate log mel spectrogram
                    log_mel_spectrogram = extract_log_mel_spectrogram(file_path, sample_rate, n_mels, target_length)
                    feature = log_mel_spectrogram.numpy().astype('float32').flatten()
                    data.append([file_index, label, feature.tolist(), log_mel_spectrogram.shape])

                    # Increment file index
                    file_index += 1

    df = pd.DataFrame(data, columns=['index', 'label', 'feature', 'shape'])

    # check if exists
    if not os.path.isfile(output_csv):
        df.to_csv(output_csv, index=False)
    else:
        df.to_csv(output_csv, mode='a', header=False, index=False)

if __name__ == "__main__":
    # paths
    # train_drone_audio_dir = "/Users/asherfish/Desktop/dataset_new/dataset8/1_train/with"
    # train_no_drone_audio_dir = "/Users/asherfish/Desktop/dataset_new/dataset8/1_train/without"
    # val_drone_audio_dir = "/Users/asherfish/Desktop/dataset_new/dataset8/2_validation/with"
    # val_no_drone_audio_dir = "/Users/asherfish/Desktop/dataset_new/dataset8/2_validation/without"
    test_drone_audio_dir = '/Users/asherfish/Desktop/dataset_new/testset/1/with'
    test_no_drone_audio_dir = '/Users/asherfish/Desktop/dataset_new/testset/1/without'

    # labels and path
    # train_audio_dirs = [train_drone_audio_dir, train_no_drone_audio_dir]
    # val_audio_dirs = [val_drone_audio_dir, val_no_drone_audio_dir]
    test_audio_dirs = [test_drone_audio_dir, test_no_drone_audio_dir]
    labels = [1, 0]  # 1 with drone，0 no drone

    # output path
    # train_output_csv = "/Users/asherfish/Desktop/dataset_new/features/dataset8/train_features.csv"
    # val_output_csv = "/Users/asherfish/Desktop/dataset_new/features/dataset8/val_features.csv"
    test_output_csv = '/Users/asherfish/Desktop/dataset_new/testset/1/test_features.csv'

    # create csv
    # process_audio_files(train_audio_dirs, labels, train_output_csv)
    # process_audio_files(val_audio_dirs, labels, val_output_csv)
    process_audio_files(test_audio_dirs, labels, test_output_csv)
