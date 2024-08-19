import os
import argparse

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms

from dataload import DFTCSVLoader
from model import Text2Energy

# Set the criterion to L1 Loss
criterion = nn.L1Loss()

def test(model, test_loader, criterion, device):
    """Evaluate the model on the test set."""
    model.eval()
    running_loss = 0.0
    outputs_list = []
    targets_list = []

    with torch.no_grad():
        for batch in test_loader:
            inputs, mass_numbers, targets = batch
            inputs = inputs.to(device)
            mass_numbers = mass_numbers.to(device)
            targets = targets.to(device)

            outputs = model(inputs, mass_numbers)
            loss = criterion(outputs.squeeze(), targets)
            running_loss += loss.item() * inputs.size(0)

            outputs_list.extend(outputs.squeeze().cpu().numpy())
            targets_list.extend(targets.cpu().numpy())

    avg_loss = running_loss / len(test_loader.dataset)
    return avg_loss, outputs_list, targets_list

def main():
    parser = argparse.ArgumentParser(description="Test the Text2Energy model.")
    parser.add_argument('--csv_file', type=str, required=True, help='Path to the CSV file.')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for data loaders.')
    parser.add_argument('--root', type=str, required=True, help='Root directory for saved models.')
    parser.add_argument('--seed', type=int, default=5, help='Random seed.')
    parser.add_argument('--input_dim', type=int, default=768, help='Input dimension.')
    parser.add_argument('--hidden_dim', type=int, default=128, help='Hidden dimension.')
    parser.add_argument('--output_dim', type=int, default=1, help='Output dimension.')
    parser.add_argument('--dropout', type=float, default=0.3, help='Dropout rate.')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the saved model.')

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    custom_transforms = transforms.Compose([transforms.ToTensor()])

    test_dataset = DFTCSVLoader(args.csv_file, transform=custom_transforms)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    model = Text2Energy(args.input_dim, args.hidden_dim, args.output_dim, args.dropout).to(device)
    model.load_state_dict(torch.load(args.model_path))
    model.eval()

    avg_loss, outputs_list, targets_list = test(model, test_loader, criterion, device)
    print(f"Test Loss: {avg_loss:.4f}")

    # Save predictions and targets
    predictions_file = os.path.join(args.root, 'predictions.txt')
    targets_file = os.path.join(args.root, 'targets.txt')

    np.savetxt(predictions_file, outputs_list)
    np.savetxt(targets_file, targets_list)

    print(f"Predictions saved to {predictions_file}")
    print(f"Targets saved to {targets_file}")

if __name__ == "__main__":
    main()
