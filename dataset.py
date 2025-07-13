import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import config


class DOADataset(Dataset):
    def __init__(self, data_path):
        data = np.load(data_path)
        self.X = data['X']
        self.Y = data['Y']

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        # Convert complex covariance matrix to real-valued tensor [2, M, M]
        cov_matrix = self.X[idx]
        real_part = torch.from_numpy(cov_matrix.real.astype(np.float32))
        imag_part = torch.from_numpy(cov_matrix.imag.astype(np.float32))

        # Flatten and create sequence of shape [M*M, 2]
        # This treats each element of the covariance matrix as a token
        model_input = torch.stack([real_part, imag_part], dim=0).view(2, -1).T

        # Prepare target sequence: [START, angle_1, ..., angle_k, END, PAD, ...]
        labels = self.Y[idx]
        labels = labels[labels != -1]  # Remove padding

        target_seq = torch.full((config.MAX_SEQ_LENGTH,), int(config.PAD_TOKEN), dtype=torch.long)
        target_seq[0] = config.START_TOKEN
        target_seq[1:len(labels) + 1] = torch.from_numpy(labels).long()
        target_seq[len(labels) + 1] = config.END_TOKEN

        return model_input, target_seq


def get_dataloader(path, batch_size, shuffle=True):
    dataset = DOADataset(path)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


if __name__ == '__main__':
    # Test the dataloader
    train_loader = get_dataloader(config.TRAIN_DATA_PATH, batch_size=4)
    for i, (src, tgt) in enumerate(train_loader):
        print("Source shape:", src.shape)  # Should be [batch, M*M, 2]
        print("Target shape:", tgt.shape)  # Should be [batch, MAX_SEQ_LENGTH]
        print("Sample target sequence:", tgt[0])
        if i == 0:
            break