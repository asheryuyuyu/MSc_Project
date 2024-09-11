import os
import librosa
import numpy as np
import soundfile as sf

def random_pitch_shift(audio, sr, n_steps_range=(-5, 5)):
    n_steps = 1
    while n_steps == 1:
        n_steps = np.random.uniform(n_steps_range[0], n_steps_range[1])
    pitch_shifted_audio = librosa.effects.pitch_shift(audio, sr, n_steps=n_steps)
    return pitch_shifted_audio

if __name__ == '__main__':
    input_dir = '/Users/asherfish/Desktop/dataset_MSc/other_noise_original'
    output_dir = '/Users/asherfish/Desktop/dataset_MSc/other_noise_augmentation'
    os.makedirs(output_dir, exist_ok=True)

    audio_files = [f for f in os.listdir(input_dir) if f.endswith('.wav')]

    count = 1
    for file_name in audio_files:

        audio_path = os.path.join(input_dir, file_name)
        audio, sr = librosa.load(audio_path, sr=None)

        pitch_shifted_audio = random_pitch_shift(audio, sr)

        output_file_name = f'pitch_{file_name}'
        output_file_path = os.path.join(output_dir, output_file_name)
        sf.write(output_file_path, pitch_shifted_audio, sr)
        count += 1
        print(f'{file_name} saved as {output_file_name}')

    print(f'DONE! totally {count} files')
