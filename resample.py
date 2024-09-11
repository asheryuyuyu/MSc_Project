import os
import librosa
import soundfile as sf

input_dir = '/Users/asherfish/Desktop/MSc/MSc_Project/dataset_1/full/scene_noise'
output_dir = '/Users/asherfish/Desktop/MSc/MSc_Project/dataset_1/full/resample_noise'

os.makedirs(output_dir, exist_ok=True)

target_sr = 16000

audio_files = [f for f in os.listdir(input_dir) if f.endswith('.wav')]

count = 1
for file_name in audio_files:
    file_path = os.path.join(input_dir, file_name)
    audio, sr = librosa.load(file_path, sr=None)

    audio_resampled = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)

    output_file_name = f'{os.path.splitext(file_name)[0]}_16khz.wav'
    output_file_path = os.path.join(output_dir, output_file_name)

    sf.write(output_file_path, audio_resampled, target_sr)
    count += 1
    print(f'{count}{file_name} saved as {output_file_name}')

print(f"ALL DONE, totally {count} files")
