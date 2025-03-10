"""
Benchmark script for testing the new medical imaging datasets (ISIC and ChestX-ray14).
This script runs a simple training loop on the datasets to verify they work correctly.
"""

import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.models as models
from data_loaders.isic import ISICDatasetWrapper
from data_loaders.chestxray import ChestXrayDatasetWrapper


def train_model(model, dataloader, criterion, optimizer, device, epochs=5):
    """Train a model on the given dataloader."""
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        
        for i, (inputs, labels) in enumerate(dataloader):
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            
            # For ChestX-ray14 (multi-label)
            if labels.dim() > 1 and labels.size(1) > 1:
                loss = criterion(outputs, labels)
                # For accuracy, we consider a prediction correct if all labels match
                predicted = (outputs > 0.5).float()
                correct += (predicted == labels).all(dim=1).sum().item()
            else:  # For ISIC (single-label)
                loss = criterion(outputs, labels)
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Statistics
            running_loss += loss.item()
            total += labels.size(0)
            
            if i % 10 == 9:  # Print every 10 mini-batches
                print(f'[Epoch {epoch + 1}, Batch {i + 1}] Loss: {running_loss / 10:.3f}, '
                      f'Accuracy: {100 * correct / total:.2f}%')
                running_loss = 0.0
        
        # Print epoch statistics
        print(f'Epoch {epoch + 1} completed. Accuracy: {100 * correct / total:.2f}%')


def evaluate_model(model, dataloader, criterion, device):
    """Evaluate a model on the given dataloader."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(inputs)
            
            # For ChestX-ray14 (multi-label)
            if labels.dim() > 1 and labels.size(1) > 1:
                loss = criterion(outputs, labels)
                # For accuracy, we consider a prediction correct if all labels match
                predicted = (outputs > 0.5).float()
                correct += (predicted == labels).all(dim=1).sum().item()
            else:  # For ISIC (single-label)
                loss = criterion(outputs, labels)
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
            
            # Statistics
            running_loss += loss.item()
            total += labels.size(0)
    
    # Print evaluation statistics
    print(f'Evaluation - Loss: {running_loss / len(dataloader):.3f}, '
          f'Accuracy: {100 * correct / total:.2f}%')


def main():
    parser = argparse.ArgumentParser(description='Benchmark medical imaging datasets')
    parser.add_argument('--dataset', type=str, choices=['isic', 'chestxray'], required=True,
                        help='Dataset to benchmark')
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to the dataset')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=5,
                        help='Number of epochs to train')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use for training (cuda or cpu)')
    
    args = parser.parse_args()
    
    # Set device
    device = torch.device(args.device)
    
    # Load dataset
    if args.dataset == 'isic':
        dataset_wrapper = ISICDatasetWrapper(args.data_path)
        num_classes = dataset_wrapper.num_cls
        criterion = nn.CrossEntropyLoss()
    elif args.dataset == 'chestxray':
        dataset_wrapper = ChestXrayDatasetWrapper(args.data_path)
        num_classes = dataset_wrapper.num_cls
        criterion = nn.BCEWithLogitsLoss()  # For multi-label classification
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")
    
    # Create data loaders
    train_loader = DataLoader(dataset_wrapper.train_dset, batch_size=args.batch_size, 
                             shuffle=True, num_workers=4)
    test_loader = DataLoader(dataset_wrapper.test_dset, batch_size=args.batch_size, 
                            shuffle=False, num_workers=4)
    
    # Create model
    model = models.resnet18(pretrained=True)
    
    # Modify the final layer based on the dataset
    if args.dataset == 'isic':
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif args.dataset == 'chestxray':
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    
    model = model.to(device)
    
    # Create optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # Train the model
    print(f"Training on {args.dataset} dataset...")
    train_model(model, train_loader, criterion, optimizer, device, epochs=args.epochs)
    
    # Evaluate the model
    print(f"Evaluating on {args.dataset} dataset...")
    evaluate_model(model, test_loader, criterion, device)


if __name__ == '__main__':
    main() 