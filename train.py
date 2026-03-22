import os
import argparse
import json
import shutil
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Subset
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def get_model(model_name, device):
    if model_name == 'resnet50':
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 2)
    elif model_name == 'inception_v3':
        model = models.inception_v3(weights=models.Inception_V3_Weights.DEFAULT, transform_input=False)
        model.aux_logits = False
        model.AuxLogits = None
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 2)
    elif model_name == 'vit_b_16':
        model = models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT)
        num_ftrs = model.heads.head.in_features
        model.heads.head = nn.Linear(num_ftrs, 2)
    else:
        raise ValueError(f"Unknown model name {model_name}")
    
    return model.to(device)

def set_parameter_requires_grad(model, freeze, model_name):
    if freeze:
        for param in model.parameters():
            param.requires_grad = False
        if model_name in ['resnet50', 'inception_v3']:
            for param in model.fc.parameters():
                param.requires_grad = True
        elif model_name == 'vit_b_16':
            for param in model.heads.head.parameters():
                param.requires_grad = True
    else:
        for param in model.parameters():
            param.requires_grad = True

def compute_class_weights(dataset):
    targets = dataset.targets if not isinstance(dataset, Subset) else [dataset.dataset.targets[i] for i in dataset.indices]
    class_counts = np.bincount(targets)
    total = len(targets)
    weights = total / (len(class_counts) * class_counts)
    return torch.FloatTensor(weights)

def run_epoch(model, dataloaders, criterion, optimizer, device):
    metrics = {'train': {}, 'val': {}}
    
    for phase in ['train', 'val']:
        if phase == 'train':
            model.train()
        else:
            model.eval()

        running_loss = 0.0
        all_preds = []
        all_labels = []

        for inputs, labels in dataloaders[phase]:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            with torch.set_grad_enabled(phase == 'train'):
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                _, preds = torch.max(outputs, 1)

                if phase == 'train':
                    loss.backward()
                    optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        epoch_loss = running_loss / len(dataloaders[phase].dataset)
        acc = accuracy_score(all_labels, all_preds)
        prec = precision_score(all_labels, all_preds, zero_division=0)
        rec = recall_score(all_labels, all_preds, zero_division=0)
        f1 = f1_score(all_labels, all_preds, zero_division=0)

        print(f"{phase.capitalize()} Loss: {epoch_loss:.4f} | Acc: {acc:.4f} | Prec: {prec:.4f} | Rec: {rec:.4f} | F1: {f1:.4f}")
        
        metrics[phase] = {'loss': epoch_loss, 'acc': acc, 'prec': prec, 'rec': rec, 'f1': f1}
        
    return metrics

def train_single_model(model_name, dataloaders, device, save_path, epochs=10, patience=3, class_weights=None):
    print(f"\n{'='*40}\nTraining {model_name.upper()}\n{'='*40}")
    model = get_model(model_name, device)
    
    if class_weights is not None:
        criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    else:
        criterion = nn.CrossEntropyLoss()
    
    # Phase 1
    print("Phase 1: Freezing base layers and training classification head...")
    set_parameter_requires_grad(model, freeze=True, model_name=model_name)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)
    
    phase1_epochs = max(1, epochs // 3)
    best_val_f1 = 0.0
    best_metrics = {}
    
    # Early stopping trackers
    epochs_no_improve = 0
    best_loss = float('inf')
    
    for epoch in range(phase1_epochs):
        print(f"Phase 1 - Epoch {epoch+1}/{phase1_epochs}")
        metrics = run_epoch(model, dataloaders, criterion, optimizer, device)
        if metrics['val']['f1'] >= best_val_f1:
            best_val_f1 = metrics['val']['f1']
            best_metrics = metrics['val']
            torch.save(model.state_dict(), save_path)
            
    # Phase 2
    print("\nPhase 2: Unfreezing all layers and fine-tuning...")
    set_parameter_requires_grad(model, freeze=False, model_name=model_name)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-5)
    
    phase2_epochs = epochs - phase1_epochs
    for epoch in range(phase2_epochs):
        print(f"Phase 2 - Epoch {epoch+1}/{phase2_epochs}")
        metrics = run_epoch(model, dataloaders, criterion, optimizer, device)
        
        val_loss = metrics['val']['loss']
        if val_loss < best_loss:
            best_loss = val_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            
        if metrics['val']['f1'] >= best_val_f1:
            best_val_f1 = metrics['val']['f1']
            best_metrics = metrics['val']
            torch.save(model.state_dict(), save_path)
            print(f"--> Saved new best model with Val F1: {best_val_f1:.4f}")
            
        if epochs_no_improve >= patience:
            print(f"Early stopping triggered after {patience} epochs with no loss improvement.")
            break

    print(f"Finished {model_name.upper()}. Best Val Metrics: {best_metrics}")
    return best_metrics

def update_metrics_json(model_name, metrics, json_path='/kaggle/working/model/metrics.json'):
    data = {}
    if os.path.exists(json_path):
        with open(json_path, 'r') as f:
            try:
                data = json.load(f)
            except:
                pass
                
    data[model_name] = metrics
    
    with open(json_path, 'w') as f:
        json.dump(data, f, indent=4)
        
    # Find best model overall out of JSON tracked records
    best_f1 = -1
    best_m = None
    for m, m_data in data.items():
        if m_data.get('f1', 0) > best_f1:
            best_f1 = m_data.get('f1', 0)
            best_m = m
            
    return best_m

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_dir = os.path.join(args.data_dir, 'Train')
    val_dir = os.path.join(args.data_dir, 'Validation')

    train_dataset = datasets.ImageFolder(train_dir, transform=transform)
    val_dataset = datasets.ImageFolder(val_dir, transform=transform)
    
    # Class weights processing
    class_weights = compute_class_weights(train_dataset)
    print(f"Computated Class Weights: {class_weights}")

    pin_mem = True if torch.cuda.is_available() else False
    dataloaders = {
        'train': DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=pin_mem),
        'val': DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=pin_mem)
    }

    out_dir = '/kaggle/working/model'
    os.makedirs(out_dir, exist_ok=True)

    save_path = os.path.join(out_dir, f"{args.model}_detector.pth")
    best_metrics = train_single_model(args.model, dataloaders, device, save_path, args.epochs, class_weights=class_weights)
    
    json_path = os.path.join(out_dir, "metrics.json")
    best_overall_model = update_metrics_json(args.model, best_metrics, json_path=json_path)
    
    print(f"\nThe current historical BEST performing model in metrics.json is {best_overall_model.upper()}")
    
    # Sync best_detector
    best_source_path = os.path.join(out_dir, f"{best_overall_model}_detector.pth")
    deployment_path = os.path.join(out_dir, 'best_detector.pth')
    if os.path.exists(best_source_path):
        shutil.copy(best_source_path, deployment_path)
        with open(os.path.join(out_dir, 'best_model_info.txt'), 'w') as f:
            f.write(best_overall_model.split('_')[0]) # Ensure it aligns with app.py naming (e.g 'resnet')
        print(f"Copied {best_source_path} to {deployment_path} for deployment.")
    
    print("Training run complete.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, choices=['resnet50', 'inception_v3', 'vit_b_16'])
    parser.add_argument('--data_dir', type=str, default='/kaggle/input/dataset', help='Kaggle dataset path')
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--batch_size', type=int, default=16)
    args = parser.parse_args()
    
    main(args)
