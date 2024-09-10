import os
import librosa
import numpy as np
import soundfile as sf


def mix_audios(drone_audio, noise_audio):
    min_len = min(len(drone_audio), len(noise_audio))
    drone_audio = drone_audio[:min_len]
    noise_audio = noise_audio[:min_len]
    mixed_audio = drone_audio + noise_audio
    max_val = np.max(np.abs(mixed_audio))
    if max_val > 0:
        mixed_audio = mixed_audio / max_val
    return mixed_audio


drone_dir = 'path_to_drone_audio_files/'
noise_dir = 'path_to_noise_audio_files/'
output_dir = 'path_to_output_audio_files/'
os.makedirs(output_dir, exist_ok=True)

drone_files = sorted([f for f in os.listdir(drone_dir) if f.endswith('.wav')])
noise_files = sorted([f for f in os.listdir(noise_dir) if f.endswith('.wav')])

assert len(drone_files) == len(noise_files), "file number not equal"
count = 1
for drone_file, noise_file in zip(drone_files, noise_files):
    drone_path = os.path.join(drone_dir, drone_file)
    drone_audio, sr_drone = librosa.load(drone_path, sr=None)

    noise_path = os.path.join(noise_dir, noise_file)
    noise_audio, sr_noise = librosa.load(noise_path, sr=None)

    assert sr_drone == sr_noise, "sample_rate not equal"

    mixed_audio = mix_audios(drone_audio, noise_audio)

    output_file_name = f'mixed_{drone_file}'
    output_file_path = os.path.join(output_dir, output_file_name)
    sf.write(output_file_path, mixed_audio, sr_drone)
    count += 1
    print(f'{count}{drone_file} and {noise_file} saved as {output_file_name}')

print(f'totally {count} files')
