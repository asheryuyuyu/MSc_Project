import os
import librosa
import soundfile as sf

# 定义源文件夹和目标文件夹路径
source_dir1 = '/Users/asherfish/Desktop/dataset_new/dataset3/drone_augmentation'
source_dir2 = '/Users/asherfish/Desktop/dataset_new/dataset3/premix_scene'
output_dir = '/Users/asherfish/Desktop/dataset_new/dataset3/mixed_drone_and_scene'

# 创建目标文件夹如果不存在
os.makedirs(output_dir, exist_ok=True)

# 获取文件夹中所有文件的列表并排序，以确保文件对齐
files1 = sorted(os.listdir(source_dir1))
files2 = sorted(os.listdir(source_dir2))

# 检查两个文件夹中文件数量是否相等
if len(files1) != len(files2):
    raise ValueError("两个文件夹中的文件数量不相等")

count = 0
# 遍历所有文件并进行重叠处理
for file1, file2 in zip(files1, files2):
    path1 = os.path.join(source_dir1, file1)
    path2 = os.path.join(source_dir2, file2)

    # 读取音频文件
    audio1, sr1 = librosa.load(path1, sr=None)
    audio2, sr2 = librosa.load(path2, sr=None)

    # 确保采样率一致
    if sr1 != sr2:
        raise ValueError(f"采样率不一致: {path1} ({sr1}), {path2} ({sr2})")

    # 确保音频长度一致
    min_length = min(len(audio1), len(audio2))
    audio1 = audio1[:min_length]
    audio2 = audio2[:min_length]

    # 重叠音频
    overlapped_audio = audio1 + audio2

    # 生成新的文件名
    base1, ext1 = os.path.splitext(file1)
    base2, ext2 = os.path.splitext(file2)
    new_filename = f"{base1}_{base2}_mixed.wav"
    output_path = os.path.join(output_dir, new_filename)

    # 保存重叠后的音频文件
    sf.write(output_path, overlapped_audio, sr1)

    count += 1
    print(f'{count} done')
print(f"{count} 音频文件重叠处理完成")
