import os
import numpy as np
import librosa
import soundfile as sf


def add_random_noise_to_audio(input_folder, output_folder, min_noise_duration=0.3, max_noise_duration=0.6, sr=16000,

    os.makedirs(output_folder, exist_ok=True)

    for file_name in os.listdir(input_folder):
        if file_name.endswith('.wav'):  
            input_path = os.path.join(input_folder, file_name)

            audio, _ = librosa.load(input_path, sr=sr)

            noise_duration = np.random.uniform(min_noise_duration, max_noise_duration)
            noise_samples = int(noise_duration * sr)
            noise = np.random.randn(noise_samples)

            noise_factor = np.random.uniform(min_noise_factor, max_noise_factor)
          
            start_position = np.random.randint(0, len(audio) - noise_samples)
          
            augmented_audio = np.copy(audio)

            augmented_audio[start_position:start_position + noise_samples] += noise_factor * noise
          
            output_file_name = f'noise_{file_name}'
            output_path = os.path.join(output_folder, output_file_name)

            sf.write(output_path, augmented_audio, sr)

            print(f"Processed {file_name} and saved to {output_path}")


if __name__ == '__main__':
    input_folder = '/Users/asherfish/Desktop/dataset_MSc/dataset_all_need/other_noise_original'
    output_folder = '/Users/asherfish/Desktop/dataset_new/dataset4/add_noise'

    add_random_noise_to_audio(input_folder, output_folder)
