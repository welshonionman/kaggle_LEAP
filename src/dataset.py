import torch
from torch.utils.data import DataLoader, Dataset

from src.constants import DEVICE


class NumpyDataset(Dataset):
    def __init__(self, x, y):
        assert x.shape[0] == y.shape[0], "Features and labels must have the same number of samples"
        self.x = x
        self.y = y
        self.device = DEVICE

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, index):
        x = torch.from_numpy(self.x[index]).float()
        y = torch.from_numpy(self.y[index]).float()
        return x, y


def get_dataloader(x_train, y_train, batch_size):
    dataset = NumpyDataset(x_train, y_train)

    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, valid_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)
    return train_loader, valid_loader
