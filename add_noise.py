import os
import numpy as np
import librosa
import soundfile as sf


def add_random_noise_to_audio(input_folder, output_folder, min_noise_duration=0.3, max_noise_duration=0.6, sr=16000,
                              min_noise_factor=0.02, max_noise_factor=0.05):
    # 确保输出文件夹存在
    os.makedirs(output_folder, exist_ok=True)

    # 遍历输入文件夹中的所有文件
    for file_name in os.listdir(input_folder):
        if file_name.endswith('.wav'):  # 仅处理 .wav 文件
            # 构建输入文件的完整路径
            input_path = os.path.join(input_folder, file_name)

            # 加载音频文件
            audio, _ = librosa.load(input_path, sr=sr)

            # 生成随机长度的噪音
            noise_duration = np.random.uniform(min_noise_duration, max_noise_duration)
            noise_samples = int(noise_duration * sr)
            noise = np.random.randn(noise_samples)

            # 生成随机噪声大小
            noise_factor = np.random.uniform(min_noise_factor, max_noise_factor)

            # 随机选择噪音插入的位置
            start_position = np.random.randint(0, len(audio) - noise_samples)

            # 创建一个副本以避免修改原始音频
            augmented_audio = np.copy(audio)

            # 在随机位置插入噪音
            augmented_audio[start_position:start_position + noise_samples] += noise_factor * noise

            # 构建输出文件的完整路径，并在文件名前添加 'noise_'
            output_file_name = f'noise_{file_name}'
            output_path = os.path.join(output_folder, output_file_name)

            # 保存带噪音的音频文件
            sf.write(output_path, augmented_audio, sr)

            print(f"Processed {file_name} and saved to {output_path}")


if __name__ == '__main__':
    input_folder = '/Users/asherfish/Desktop/dataset_MSc/dataset_all_need/other_noise_original'
    output_folder = '/Users/asherfish/Desktop/dataset_new/dataset4/add_noise'

    add_random_noise_to_audio(input_folder, output_folder)
