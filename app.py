import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from flask import Flask, request, render_template, jsonify
from PIL import Image
import io
import base64

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

model_path = "model/best_detector.pth"
info_path = "model/best_model_info.txt"
model = None
model_name = "Unknown"

def get_model_architecture(name):
    if name == 'resnet':
        m = models.resnet50()
        m.fc = nn.Linear(m.fc.in_features, 2)
    elif name == 'inception':
        m = models.inception_v3(transform_input=False)
        m.aux_logits = False
        m.AuxLogits = None
        m.fc = nn.Linear(m.fc.in_features, 2)
    elif name == 'vit':
        m = models.vit_b_16()
        m.heads.head = nn.Linear(m.heads.head.in_features, 2)
    else:
        raise ValueError(f"Unknown architecture {name}")
    return m

def load_model():
    global model, model_name
    if os.path.exists(model_path) and os.path.exists(info_path):
        try:
            with open(info_path, 'r') as f:
                model_name = f.read().strip()
            
            model = get_model_architecture(model_name)
            model.load_state_dict(torch.load(model_path, map_location=device))
            model = model.to(device)
            model.eval()
            print(f"Model ({model_name}) loaded successfully for deployment.")
        except Exception as e:
            print(f"Error loading model weights: {e}")
            model = None
    else:
        print(f"Warning: Model file not found. Please run train.py first.")

load_model()

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
        
    try:
        image_bytes = file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        encoded_img = base64.b64encode(image_bytes).decode('utf-8')
        
        if model is None:
            return jsonify({
                'error': 'Model not trained yet.',
                'image': encoded_img
            })
            
        img_t = transform(image)
        batch_t = torch.unsqueeze(img_t, 0).to(device)
        
        with torch.no_grad():
            outputs = model(batch_t)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            
            fake_prob = probabilities[0][0].item() * 100
            real_prob = probabilities[0][1].item() * 100
            
            if real_prob > fake_prob:
                result = "Real"
                confidence = real_prob
            else:
                result = "Fake"
                confidence = fake_prob
                
        return jsonify({
            'success': True,
            'result': result,
            'confidence': f"{confidence:.2f}%",
            'real_percentage': f"{real_prob:.2f}%",
            'fake_percentage': f"{fake_prob:.2f}%",
            'architecture': model_name.upper(),
            'image': encoded_img
        })

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True, port=5000)
