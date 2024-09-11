import os
import librosa
import numpy as np
import soundfile as sf


def random_time_stretch(audio, sr, target_length, stretch_range=(0.5, 2)):
    stretch_factor = 1.0
    while stretch_factor == 1.0:
        stretch_factor = np.random.uniform(stretch_range[0], stretch_range[1])
    stretched_audio = librosa.effects.time_stretch(audio, rate=stretch_factor)

    if len(stretched_audio) < target_length:
        padding = np.zeros(target_length - len(stretched_audio))
        stretched_audio = np.concatenate((stretched_audio, padding))
    else:
        stretched_audio = stretched_audio[:target_length]

    return stretched_audio

if __name__ == '__main__':
    input_dir = '/Users/asherfish/Desktop/dataset_MSc/other_noise_original'
    output_dir = '/Users/asherfish/Desktop/dataset_MSc/other_noise_augmentation'
    os.makedirs(output_dir, exist_ok=True)

    audio_files = [f for f in os.listdir(input_dir) if f.endswith('.wav')]

    count = 1
    for file_name in audio_files:

        audio_path = os.path.join(input_dir, file_name)
        audio, sr = librosa.load(audio_path, sr=None)

        target_length = len(audio)
        stretched_audio = random_time_stretch(audio, sr, target_length)

        output_file_name = f'speed_change_{file_name}'
        output_file_path = os.path.join(output_dir, output_file_name)
        sf.write(output_file_path, stretched_audio, sr)
        count += 1
        print(f'{file_name} saved as {output_file_name}')

    print(f'DONE! totally {count} files')
