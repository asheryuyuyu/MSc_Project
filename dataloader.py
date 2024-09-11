import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class AudioDataset(Dataset):
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        feature_str = self.data.iloc[idx, 2]
        feature = np.fromstring(feature_str.strip('[]'), sep=',', dtype=np.float32)
        feature = feature.reshape(1, 40, 51)
        label = int(self.data.iloc[idx, 1])
        return torch.tensor(feature), torch.tensor(label)


def load_data(train_csv, val_csv, batch_size=64):
    train_dataset = AudioDataset(train_csv)
    val_dataset = AudioDataset(val_csv)
    #test_dataset = AudioDataset(test_csv)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    #test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader


