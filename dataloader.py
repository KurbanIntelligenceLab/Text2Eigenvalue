import numpy as np
import pandas as pd
import torch
from ast import literal_eval
from torch.utils.data import Dataset


class Text2EverythingLoader(Dataset):
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

class QMDataloader(Dataset):
    def __init__(self, csv_file, label_column, feature_columns, normalize_labels=True):
        # Load the data from CSV
        self.data = pd.read_csv(csv_file, low_memory=False)

        # drop nan text embeddings
        self.data = self.data.dropna(subset=['text_embedding'])

        # Parse the text embeddings with robust handling
        self.data['text_embedding'] = self.data['text_embedding'].apply(self.clean_and_parse_embedding)

        # Filter out rows where text_embedding is None
        self.data = self.data[self.data['text_embedding'].apply(lambda x: x is not None)]

        # Convert text embeddings into PyTorch tensors
        self.text_embeddings = [torch.Tensor(x) for x in self.data['text_embedding']]

        # Prepare other features and labels
        self.features = torch.Tensor(self.data[feature_columns].values)
        self.labels = torch.Tensor(self.data[label_column].values)

        # Normalize labels if required
        if normalize_labels:
            self.label_mean = self.labels.mean()
            self.label_std = self.labels.std()
            self.labels = (self.labels - self.label_mean) / self.label_std
        else:
            self.label_mean = 0
            self.label_std = 1

    def clean_and_parse_embedding(self, embedding_str):
        try:
            # Remove square brackets and split by commas
            embedding_list = [float(item) for item in embedding_str.strip('[]').split(',')]
            return embedding_list
        except (ValueError, SyntaxError) as e:
            # If parsing fails, return None to filter out later
            print(f"Failed to parse embedding: {embedding_str}")
            return None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        features = self.features[idx]
        label = self.labels[idx]
        text_embedding = self.text_embeddings[idx]
        return features, text_embedding, label

    def denormalize_label(self, normalized_label):
        return normalized_label * self.label_std + self.label_mean

