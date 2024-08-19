import numpy as np
import pandas as pd
import torch
from ast import literal_eval
from torch.utils.data import Dataset


class Text2EnergyLoader(Dataset):
    def __init__(self, csv_file, transform=None):
        self.data = pd.read_csv(csv_file)
        self.data['textStringvector'] = self.data['textStringvector'].apply(literal_eval)
        self.text_vectors = [torch.Tensor(x) for x in self.data['textStringvector']]
        self.mass_numbers = torch.Tensor(self.data['mass_number'].values).unsqueeze(1)
        self.total_energy = torch.Tensor(self.data['total_energy'].values)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        text_vector = self.text_vectors[index]
        mass_number = self.mass_numbers[index]
        total_energy = self.total_energy[index]

        if self.transform:
            text_vector = self.transform(text_vector)

        return text_vector, mass_number, total_energy


class VectorAugmentation:
    def __init__(self, noise_std=0.1, shift_range=5, augmentation_prob=0.5):
        self.noise_std = noise_std
        self.shift_range = shift_range
        self.augmentation_prob = augmentation_prob

    def __call__(self, vector):
        if np.random.uniform() < self.augmentation_prob:
            vector = self.apply_noise(vector)
            vector = self.apply_shift(vector)
        return vector

    def apply_noise(self, vector):
        noise = torch.randn_like(vector) * self.noise_std
        augmented_vector = vector + noise
        return augmented_vector

    def apply_shift(self, vector):
        shift_amount = np.random.randint(-self.shift_range, self.shift_range + 1)
        augmented_vector = torch.roll(vector, shift_amount)
        return augmented_vector

# Example usage:
# dataset = DFTCSVLoader(csv_file='data.csv', transform=VectorAugmentation(noise_std=0.1, shift_range=5, augmentation_prob=0.5))
# dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
