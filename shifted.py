import os
import librosa
import numpy as np
import soundfile as sf

# random shift
def time_shift(audio, sr, shift_max_seconds=0.5):
    shift = 0
    while shift == 0:
        shift = np.random.randint(1, int(shift_max_seconds * sr))
    direction = np.random.choice([-1, 1])
    if direction == -1:
        shifted_audio = np.pad(audio, (shift, 0), mode='constant')[:len(audio)]
    else:
        shifted_audio = np.pad(audio, (0, shift), mode='constant')[-len(audio):]
    return shifted_audio

'''

# mixing
def mix_audios(audio, noise):
    min_len = min(len(audio), len(noise))
    audio = audio[:min_len]
    noise = noise[:min_len]
    mixed_audio = audio + noise
    max_val = np.max(np.abs(mixed_audio))
    if max_val > 0:
        mixed_audio = mixed_audio / max_val
    return mixed_audio
'''

if __name__ == '__main__':

    input_dir = '/Users/asherfish/Desktop/dataset_MSc/other_noise_original'
    output_dir = '/Users/asherfish/Desktop/dataset_MSc/other_noise_augmentation'
    os.makedirs(output_dir, exist_ok=True)

    #noise, sr_noise = librosa.load(noise_file, sr=None)
    audio_files = [f for f in os.listdir(input_dir) if f.endswith('.wav')]

    count = 1
    for file_name in audio_files:

        audio_path = os.path.join(input_dir, file_name)
        audio, sr = librosa.load(audio_path, sr=None)

        shifted_audio = time_shift(audio, sr)

        #mixed_audio = mix_audios(shifted_audio, noise)

        output_file_name = f'shifted_{file_name}'
        output_file_path = os.path.join(output_dir, output_file_name)
        sf.write(output_file_path, shifted_audio, sr)

        count += 1
        print(f'{file_name} saved as {output_file_name}')

    print(f"ALL DONEEEEEE! totally {count} files")
