
# Molecule Property Prediction with Text2Everything

This official repository contains scripts for training and testing the `Text2Everything` model, a deep learning model for predicting molecule properties from textual representations.

## Requirements

- Python 3.9+
- PyTorch
- torchvision
- numpy
- pandas
- Pillow
- tqdm
- matplotlib

## Setup

1. Clone the repository:

    ```bash
    git clone git@github.com:KurbanIntelligenceLab/Text2Everything.git
    cd Text2Everything
    ```

2. Install the required packages:

    ```bash
    pip install -r requirements.txt
    ```
## Data

You can download the data for elements and molecules from the following link: [Text2Everything Data](https://tamucs-my.sharepoint.com/:f:/r/personal/hasan_kurban_tamu_edu/Documents/KIL-OneDrive/Can%20Polat/Text2Everything/data?csf=1&web=1&e=WfhepS)

## Usage

### Training

The training script trains the `Text2Everything` model and validates it during training. It supports various configurations through command-line arguments.

#### Arguments

- `--csv_file`: Path to the CSV file (required).
- `--num_epochs`: Number of training epochs (default: 15000).
- `--learning_rate`: Learning rate for the optimizer (default: 0.001).
- `--batch_size`: Batch size for data loaders (default: 64).
- `--root`: Root directory for saving results (required).
- `--seed`: Random seed (default: 5).
- `--input_dim`: Input dimension (default: 768).
- `--hidden_dim`: Hidden dimension (default: 128).
- `--output_dim`: Output dimension (default: 1).
- `--dropout`: Dropout rate (default: 0.3).
- `--val_split`: Validation split ratio (default: 0.2).
- `--augmentation_prob`: Probability of applying augmentation (default: 0.7).

#### Example Command

```bash
python train.py --csv_file "data.csv" --root "/path/to/root"
```

### Testing

The test script evaluates the `Text2Everything` model on a test dataset.

#### Arguments

- `--csv_file`: Path to the CSV file (required).
- `--batch_size`: Batch size for data loaders (default: 64).
- `--root`: Root directory for saving results (required).
- `--seed`: Random seed (default: 5).
- `--input_dim`: Input dimension (default: 768).
- `--hidden_dim`: Hidden dimension (default: 128).
- `--output_dim`: Output dimension (default: 1).
- `--dropout`: Dropout rate (default: 0.3).
- `--model_path`: Path to the saved model (required).

#### Example Command

```bash
python test.py --csv_file "test_data.csv" --root "/path/to/results" --model_path "/path/to/saved_model.pth"
```

### Results

The scripts will save the following results:
- **Training**: The best model and final model will be saved in the specified root directory along with the training and validation loss plots.
- **Testing**: The predictions and targets will be saved in the specified root directory.

### Directory Structure

Example directory structure for saved results:
```
results/
├── train/
│   ├── best_model.pth
│   ├── final_model.pth
│   ├── loss.png
├── test/
│   ├── predictions.txt
│   ├── targets.txt
```
