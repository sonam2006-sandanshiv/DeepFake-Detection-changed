import os
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score
from train import get_model

def load_saved_model(model_name, path, device):
    if not os.path.exists(path):
        print(f"File {path} not found. Skipping {model_name}.")
        return None
    model = get_model(model_name, device)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    return model

def main(data_dir='/kaggle/input/dataset'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluation utilizing device: {device}")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    val_dir = os.path.join(data_dir, 'Validation')
    if not os.path.exists(val_dir):
        print(f"Validation dataset not found at {val_dir}. Exiting.")
        return

    pin_mem = True if torch.cuda.is_available() else False
    val_dataset = datasets.ImageFolder(val_dir, transform=transform)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=2, pin_memory=pin_mem)
    
    base_dir = '/kaggle/working/model/'
    resnet = load_saved_model('resnet50', os.path.join(base_dir, 'resnet50_detector.pth'), device)
    efficientnet = load_saved_model('efficientnet', os.path.join(base_dir, 'efficientnet_detector.pth'), device)
    mobilenet = load_saved_model('mobilenet', os.path.join(base_dir, 'mobilenet_detector.pth'), device)
    
    active_models = {}
    if resnet: active_models['resnet50'] = resnet
    if efficientnet: active_models['efficientnet'] = efficientnet
    if mobilenet: active_models['mobilenet'] = mobilenet
    
    if len(active_models) == 0:
        print("No models found to evaluate.")
        return

    print(f"\nEvaluating models offline: {list(active_models.keys())}")
    
    all_labels = []
    model_preds = {name: [] for name in active_models.keys()}
    ensemble_preds = []

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            all_labels.extend(labels.numpy())
            
            batch_logits = []
            
            for name, model in active_models.items():
                outputs = model(inputs) # Logits before Softmax
                
                _, preds = torch.max(outputs, 1)
                model_preds[name].extend(preds.cpu().numpy())
                batch_logits.append(outputs.cpu())
                
            # Ensemble logic: Average LOGITS for Better Stability
            if len(batch_logits) > 0:
                avg_logits = torch.mean(torch.stack(batch_logits), dim=0)
                _, ens_preds = torch.max(avg_logits, 1)
                ensemble_preds.extend(ens_preds.numpy())

    print("\n========= REPORT =========")
    for name, preds in model_preds.items():
        acc = accuracy_score(all_labels, preds)
        f1 = f1_score(all_labels, preds, zero_division=0)
        print(f"Single Model [{name.upper()}]: Acc = {acc:.4f}, F1 = {f1:.4f}")

    if len(active_models) > 1:
        ens_acc = accuracy_score(all_labels, ensemble_preds)
        ens_f1 = f1_score(all_labels, ensemble_preds, zero_division=0)
        print(f"ENSEMBLE (Average Logits Pooling): Acc = {ens_acc:.4f}, F1 = {ens_f1:.4f}")
        print("==========================\n")
        print("Note: The Kaggle deployment recommends only deploying the single best configuration for production.")

if __name__ == '__main__':
    main()
