import os
import librosa
import soundfile as sf
import numpy as np


# Define the normalization function
def normalize_to_range(audio, target_min, target_max):
    min_val = np.min(audio)
    max_val = np.max(audio)
    norm_audio = (audio - min_val) / (max_val - min_val)
    return norm_audio * (target_max - target_min) + target_min


# def max_abs_normalization(audio):
#     return audio / np.max(np.abs(audio))
#
# def min_max_normalization(audio):
#     min_val = np.min(audio)
#     max_val = np.max(audio)
#     return (audio - min_val) / (max_val - min_val)

if __name__ == '__main__':
    # Define the source and target directories
    source_dir = '/Users/asherfish/Desktop/dataset_new/testset/2/without'
    target_dir = '/Users/asherfish/Desktop/dataset_new/testset/2/without_normalised'

    # Ensure the target directory exists
    os.makedirs(target_dir, exist_ok=True)
    i = 0
    # Process each audio file in the source directory
    for filename in os.listdir(source_dir):
        if filename.endswith('.wav'):
            # Load the audio file
            file_path = os.path.join(source_dir, filename)
            audio, sample_rate = librosa.load(file_path, sr=None)

            # Normalize the audio
            audio_normalized = normalize_to_range(audio, -1, 1)
            # audio_normalized = max_abs_normalization(audio)
            # audio_normalized = min_max_normalization(audio)

            # Create the new filename with '_norm' suffix
            base, ext = os.path.splitext(filename)
            new_filename = f"{base}_norm{ext}"

            # Save the normalized audio to the target directory
            target_file_path = os.path.join(target_dir, new_filename)
            sf.write(target_file_path, audio_normalized, sample_rate)
        i += 1
        print(f'{i}')
    print(f"{i} Processing completed.")
