import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Subset
import numpy as np

def train_model(data_dir, epochs=1, batch_size=32, sample_size=None, model_path="model/deepfake_detector.pth"):
    # Determine device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Data transformation
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    print("Loading dataset...")
    # Load dataset
    train_dir = os.path.join(data_dir, 'Train')
    val_dir = os.path.join(data_dir, 'Validation')

    train_dataset = datasets.ImageFolder(train_dir, transform=transform)
    val_dataset = datasets.ImageFolder(val_dir, transform=transform)

    if sample_size:
        # Get subset of data for quick training
        print(f"Using subset of {sample_size} per class for quick training...")
        train_indices = []
        for i in range(len(train_dataset.classes)):
            class_indices = np.where(np.array(train_dataset.targets) == i)[0]
            np.random.shuffle(class_indices)
            train_indices.extend(class_indices[:sample_size])
        train_dataset = Subset(train_dataset, train_indices)
        
        val_indices = []
        for i in range(len(val_dataset.classes)):
            class_indices = np.where(np.array(val_dataset.targets) == i)[0]
            np.random.shuffle(class_indices)
            val_indices.extend(class_indices[:int(sample_size * 0.2)])
        val_dataset = Subset(val_dataset, val_indices)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    class_names = train_dataset.dataset.classes if isinstance(train_dataset, Subset) else train_dataset.classes
    print(f"Classes: {class_names}")

    # Load pre-trained MobileNetV2 for fast training
    print("Initializing model...")
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
    
    # Freeze earlier layers for faster training
    for param in model.parameters():
        param.requires_grad = False
        
    # Replace classifier
    num_ftrs = model.classifier[1].in_features
    # Binary classification
    model.classifier[1] = nn.Linear(num_ftrs, 2)
    
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)

    print("Starting training...")
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        print("-" * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
                dataloader = train_loader
            else:
                model.eval()   # Set model to evaluate mode
                dataloader = val_loader

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloader.dataset)
            epoch_acc = running_corrects.double() / len(dataloader.dataset)

            print(f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

    print("Training complete.")
    
    # Ensure model dir exists
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    
    # Save the model
    print(f"Saving model to {model_path}...")
    torch.save(model.state_dict(), model_path)
    print("Model saved successfully.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Deepfake Detection Model')
    parser.add_argument('--data_dir', type=str, default='c:/Users/sonam/Downloads/archive/Dataset', help='Path to dataset')
    parser.add_argument('--epochs', type=int, default=1, help='Number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--sample_size', type=int, default=500, help='Number of images per class to use for quick training (None for all)')
    
    args = parser.parse_args()
    
    train_model(args.data_dir, epochs=args.epochs, batch_size=args.batch_size, sample_size=args.sample_size)
