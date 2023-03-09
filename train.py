import os
import csv
import argparse
import yaml
import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

from models import get_model
from utils import AverageMeter, compute_mse, compute_mae, compute_psnr, load_state_dict

class CrowdCountingDataset(data.Dataset):
    """Custom dataset class for crowd counting."""

    def __init__(self, file_list, transform=None):
        self.file_list = file_list
        self.transform = transform

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        # Load image and density map data
        filename = self.file_list[index]
        image = Image.open(filename).convert('RGB')
        density_map_file = os.path.splitext(filename)[0] + '.h5'
        with h5py.File(density_map_file, 'r') as f:
            density_map = np.array(f['density'])
        
        # Apply data transforms
        if self.transform:
            image = self.transform(image)
            density_map = self.transform(density_map)

        return image, density_map

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True, help='path to dataset YAML file')
    parser.add_argument('--epochs', type=int, default=10, help='number of epochs to train the model')
    parser.add_argument('--batch-size', type=int, default=16, help='batch size for training and validation')
    return parser.parse_args()



def main():
    # Parse command line arguments
    args = parse_arguments()

    # Load dataset information from YAML file
    with open(args.dataset, 'r') as f:
        dataset_info = yaml.load(f, Loader=yaml.Loader)

    # Define transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Define dataset
    train_dataset = ImageFolder(dataset_info['train_file_list'], transform=transform)
    val_dataset = ImageFolder(dataset_info['val_file_list'], transform=transform)

    # Define data loader
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)

    # Define loss function and optimizer
    criterion = nn.MSELoss()
    model = get_model()  # Replace this with your own model
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Move model to GPU if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Create result folder if it doesn't exist
    if not os.path.exists('result'):
        os.makedirs('result')

    # Define CSV writer to save evaluation results
    result_file = os.path.join('result', 'results.csv')
    with open(result_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Epoch', 'Train Loss', 'Val Loss', 'MSE', 'MAE', 'PSNR'])

        # Train the model
        for epoch in range(args.epochs):
            train_loss_meter = AverageMeter()
            val_loss_meter = AverageMeter()
            mse_meter = AverageMeter()
            mae_meter = AverageMeter()
            psnr_meter = AverageMeter()

            # Train the model
            model.train()
            for images, targets in train_loader:
                images = images.to(device)
                targets = targets.to(device)
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                train_loss_meter.update(loss.item(), images.size(0))

            # Validate the model
            model.eval()
            with torch.no_grad():
                for images, targets in val_loader:
                    images = images.to(device)
                    targets = targets.to(device)
                    outputs = model(images)
                    loss = criterion(outputs, targets)
                    val_loss_meter.update(loss.item(), images.size(0))

                    # Evaluate MSE, MAE, and PSNR on validation set
                    for images, targets in val_loader:
                        images = images.to(device)
                        targets = targets.to(device)

                        # Forward pass
                        outputs = model(images)

                        # Compute MSE and MAE
                        mse = criterion(outputs, targets)
                        mae = torch.abs(outputs - targets).mean()

                        # Compute PSNR
                        psnr = 10 * torch.log10(1 / mse)

                        mse_meter.update(mse.item(), images.size(0))
                        mae_meter.update(mae.item(), images.size(0))
                        psnr_meter.update(psnr.item(), images.size(0))
    # Print and save results
    result = [epoch+1, train_loss_meter.avg, val_loss_meter.avg, mse_meter.avg, mae_meter.avg, psnr_meter.avg]
    print(f'Epoch {epoch+1}: Train Loss: {train_loss_meter.avg:.4f}, Val Loss: {val_loss_meter.avg:.4f}, MSE: {mse_meter.avg:.4f}, MAE: {mae_meter.avg:.4f}, PSNR: {psnr_meter.avg:.4f}')

    # Save results to CSV
    with open(os.path.join(result_dir, 'results.csv'), 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(result)

    # Save model weights
    torch.save(model.state_dict(), os.path.join(result_dir, f'weights_epoch{epoch+1}.pt'))
