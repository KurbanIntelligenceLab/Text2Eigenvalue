import argparse
import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim
from dataloader import Text2EnergyLoader, VectorAugmentation
from torch.utils.data import DataLoader, SubsetRandomSampler
from tqdm import tqdm

from model import Text2Energy


def reset_weights(model):
    for layer in model.children():
        if hasattr(layer, 'reset_parameters'):
            layer.reset_parameters()

def train(model, train_loader, optimizer, criterion, device, augmentation_transform=None):
    model.train()
    running_loss = 0.0

    for batch in train_loader:
        inputs, mass_numbers, targets = batch
        if augmentation_transform:
            inputs = augmentation_transform(inputs)
        optimizer.zero_grad()
        outputs = model(inputs.to(device), mass_numbers.to(device))
        loss = criterion(outputs.squeeze(), targets.to(device))
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)

    return running_loss / len(train_loader.dataset)

def validate(model, val_loader, criterion, device):
    model.eval()
    running_loss = 0.0

    with torch.no_grad():
        for batch in val_loader:
            inputs, mass_numbers, targets = batch
            outputs = model(inputs.to(device), mass_numbers.to(device))
            loss = criterion(outputs.squeeze(), targets.to(device))
            running_loss += loss.item() * inputs.size(0)

    return running_loss / len(val_loader.dataset)

def main():
    parser = argparse.ArgumentParser(description="Train and validate the Text2Energy model.")
    parser.add_argument('--csv_file', type=str, required=True, help='Path to the CSV file.')
    parser.add_argument('--num_epochs', type=int, default=15000, help='Number of training epochs.')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for the optimizer.')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for data loaders.')
    parser.add_argument('--root', type=str, required=True, help='Root directory for saving results.')
    parser.add_argument('--seed', type=int, default=5, help='Random seed.')
    parser.add_argument('--input_dim', type=int, default=768, help='Input dimension.')
    parser.add_argument('--hidden_dim', type=int, default=128, help='Hidden dimension.')
    parser.add_argument('--output_dim', type=int, default=1, help='Output dimension.')
    parser.add_argument('--dropout', type=float, default=0.3, help='Dropout rate.')
    parser.add_argument('--val_split', type=float, default=0.2, help='Validation split ratio.')
    parser.add_argument('--augmentation_prob', type=float, default=0.7, help='Probability of applying augmentation.')

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataset = Text2EnergyLoader(args.csv_file, transform=None)
    val_split = args.val_split
    num_train = len(dataset)
    indices = list(range(num_train))
    split = int(np.floor(val_split * num_train))
    np.random.shuffle(indices)
    train_idx, val_idx = indices[split:], indices[:split]

    train_sampler = SubsetRandomSampler(train_idx)
    val_sampler = SubsetRandomSampler(val_idx)

    train_loader = DataLoader(dataset, batch_size=args.batch_size, sampler=train_sampler)
    val_loader = DataLoader(dataset, batch_size=args.batch_size, sampler=val_sampler)

    model = Text2Energy(args.input_dim, args.hidden_dim, args.output_dim, args.dropout).to(device)
    reset_weights(model)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=0.0001)

    augmentation_transform = VectorAugmentation(noise_std=0.3, shift_range=7, augmentation_prob=args.augmentation_prob)

    train_loss_values = []
    val_loss_values = []
    best_val_loss = float('inf')

    for epoch in tqdm(range(args.num_epochs)):
        train_loss = train(model, train_loader, optimizer, criterion, device, augmentation_transform)
        val_loss = validate(model, val_loader, criterion, device)

        train_loss_values.append(train_loss)
        val_loss_values.append(val_loss)

        print(f"Epoch {epoch+1}/{args.num_epochs}, Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(args.root, 'best_model.pth'))
            print("Saved best model!")

if __name__ == "__main__":
    main()
