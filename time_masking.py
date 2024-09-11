import os
import librosa
import numpy as np
import soundfile as sf


def random_time_mask(audio, sr, mask_duration_range=(0.1, 0.5), num_masks=1):
    audio_length = len(audio)
    for _ in range(num_masks):
        mask_duration = np.random.uniform(mask_duration_range[0], mask_duration_range[1]) * sr
        mask_start = np.random.randint(0, audio_length - int(mask_duration))
        audio[mask_start:mask_start + int(mask_duration)] = 0
    return audio

if __name__ == '__main__':
    input_dir = '/Users/asherfish/Desktop/dataset_MSc/other_noise_original'
    output_dir = '/Users/asherfish/Desktop/dataset_MSc/other_noise_augmentation'
    os.makedirs(output_dir, exist_ok=True)

    audio_files = [f for f in os.listdir(input_dir) if f.endswith('.wav')]

    count = 1
    for file_name in audio_files:
        audio_path = os.path.join(input_dir, file_name)
        audio, sr = librosa.load(audio_path, sr=None)

        masked_audio = random_time_mask(audio, sr)

        output_file_name = f'masked_{file_name}'
        output_file_path = os.path.join(output_dir, output_file_name)
        sf.write(output_file_path, masked_audio, sr)
        count += 1
        print(f'{file_name} saved as {output_file_name}')

    print(f'DONE! totally {count} files')
