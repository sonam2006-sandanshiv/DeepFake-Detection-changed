import os
import torch
import torch.nn as nn
import numpy as np
from PIL import Image, ImageFilter
from torchvision import models, transforms
from flask import Flask, request, render_template, jsonify
from flask_cors import CORS
import io
import base64
try:
    from transformers import CLIPProcessor, CLIPModel
except ImportError:
    CLIPProcessor, CLIPModel = None, None

app = Flask(__name__)
CORS(app)  # Allow cross-origin from Vercel frontend
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

models_ensemble = {}

def get_model_architecture(name):
    if name in ['resnet', 'resnet50']:
        m = models.resnet50()
        m.fc = nn.Linear(m.fc.in_features, 2)
    elif name == 'efficientnet':
        m = models.efficientnet_b0()
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, 2)
    elif name == 'mobilenet':
        m = models.mobilenet_v3_large()
        m.classifier[3] = nn.Linear(m.classifier[3].in_features, 2)
    else:
        raise ValueError(f"Unknown architecture {name}")
    return m

def _maybe_download_model(dest_path, env_key):
    """Download model from URL in env var if file doesn't exist."""
    if os.path.exists(dest_path):
        return True
    url = os.environ.get(env_key)
    if not url:
        print(f"No env var {env_key} set and {dest_path} missing — skipping.")
        return False
    try:
        import urllib.request
        print(f"Downloading {dest_path} from {url}...")
        os.makedirs(os.path.dirname(dest_path) or '.', exist_ok=True)
        urllib.request.urlretrieve(url, dest_path)
        print(f"Downloaded {dest_path}.")
        return True
    except Exception as e:
        print(f"Download failed for {dest_path}: {e}")
        return False

def load_models():
    global models_ensemble
    base_dir = "model"
    os.makedirs(base_dir, exist_ok=True)
    
    model_files = {
        'resnet50':     ('resnet50_detector.pth',    'HF_RESNET_MODEL'),
        'efficientnet': ('efficientnet_detector.pth', 'HF_EFFICIENTNET_MODEL'),
        'mobilenet':    ('mobilenet_detector.pth',   'HF_MOBILENET_MODEL'),
    }
    
    for name, (filename, env_key) in model_files.items():
        path = os.path.join(base_dir, filename)
        _maybe_download_model(path, env_key)
        if os.path.exists(path):
            try:
                m = get_model_architecture(name)
                m.load_state_dict(torch.load(path, map_location=device))
                m = m.to(device)
                m.eval()
                models_ensemble[name] = m
                print(f"Loaded {name} successfully.")
            except Exception as e:
                print(f"Error loading {name}: {e}")
        else:
            print(f"Warning: {path} not found.")

clip_model = None
clip_processor = None

def load_clip():
    global clip_model, clip_processor
    if CLIPModel is None:
        print("Transformers not found, skipping CLIP.")
        return
        
    try:
        print("Loading CLIP ViT-B/32...")
        clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32", local_files_only=False)
        clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32", local_files_only=False)
        clip_model = clip_model.to(device)
        clip_model.eval()
        print("CLIP loaded successfully.")
    except Exception as e:
        print(f"Error loading CLIP: {e}")

load_models()
load_clip()

def compute_analysis(image):
    """Heuristic image analysis — returns 5 signals in 0-100 range."""
    img = image.convert('RGB').resize((256, 256))
    arr = np.array(img, dtype=np.float32)

    # 1. Facial Asymmetry — pixel diff between left and right halves
    left  = arr[:, :128, :]
    right = np.fliplr(arr[:, 128:, :])
    asym = float(np.mean(np.abs(left - right))) / 255.0 * 100
    asym = min(asym * 3.5, 100.0)  # scale to readable range

    # 2. Colour Irregularity — standard deviation of HSV saturation channel
    hsv = image.convert('HSV').resize((256, 256))
    sat = np.array(hsv, dtype=np.float32)[:, :, 1]
    colour = float(np.std(sat)) / 128.0 * 100
    colour = min(colour * 1.5, 100.0)

    # 3. Digital Artifacts — high-frequency Laplacian energy
    gray = img.mean(axis=2)
    laplacian = np.array(image.convert('L').resize((256, 256)).filter(ImageFilter.FIND_EDGES), dtype=np.float32)
    artifacts = float(np.mean(laplacian)) / 255.0 * 100
    artifacts = min(artifacts * 2.5, 100.0)

    # 4. Blur Inconsistency — variance of local Laplacian across 4 quadrants
    pil_gray = image.convert('L').resize((256, 256))
    quad_vars = []
    for r in [(0,0,128,128),(128,0,256,128),(0,128,128,256),(128,128,256,256)]:
        q = np.array(pil_gray.crop(r).filter(ImageFilter.FIND_EDGES), dtype=np.float32)
        quad_vars.append(float(np.var(q)))
    blur = float(np.std(quad_vars)) / 500.0 * 100
    blur = min(blur * 2.0, 100.0)

    # 5. Noise Pattern — residual noise after blurring
    smooth = pil_gray.filter(ImageFilter.GaussianBlur(radius=2))
    noise_arr = np.abs(np.array(pil_gray, dtype=np.float32) - np.array(smooth, dtype=np.float32))
    noise = float(np.mean(noise_arr)) / 30.0 * 100
    noise = min(noise, 100.0)

    return {
        'asymmetry': round(asym, 1),
        'colour':    round(colour, 1),
        'artifacts': round(artifacts, 1),
        'blur':      round(blur, 1),
        'noise':     round(noise, 1),
    }

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
        
        if len(models_ensemble) == 0:
            return jsonify({
                'error': 'No models loaded. Please configure HF_RESNET_MODEL, HF_EFFICIENTNET_MODEL, HF_MOBILENET_MODEL env vars on Render.',
                'image': encoded_img
            })
            
        img_t = transform(image)
        batch_t = torch.unsqueeze(img_t, 0).to(device)
        
        with torch.no_grad():
            out_eff = models_ensemble['efficientnet'](batch_t)
            out_res = models_ensemble['resnet50'](batch_t)
            out_mob = models_ensemble['mobilenet'](batch_t)
            
            # Apply softmax FIRST to get individual model probabilities
            prob_eff = torch.nn.functional.softmax(out_eff, dim=1)
            prob_res = torch.nn.functional.softmax(out_res, dim=1)
            prob_mob = torch.nn.functional.softmax(out_mob, dim=1)
            
            # Weighted ensemble of probabilities (much more robust than logits)
            probabilities = (0.5 * prob_eff + 0.3 * prob_res + 0.2 * prob_mob)
            
            fake_prob = probabilities[0][0].item()
            real_prob = probabilities[0][1].item()
            
            # CLIP Semantic Analysis
            clip_real_prob = 0.0
            clip_ai_prob = 0.0
            clip_edited_prob = 0.0
            clip_fake_prob = 0.0
            
            if clip_model and clip_processor:
                prompts = ["a real photograph", "an AI-generated image", "a manipulated or edited image"]
                inputs = clip_processor(text=prompts, images=image, return_tensors="pt", padding=True)
                
                # Move inputs to device
                for k, v in inputs.items():
                    if isinstance(v, torch.Tensor):
                        inputs[k] = v.to(device)
                        
                clip_outputs = clip_model(**inputs)
                clip_probs = clip_outputs.logits_per_image.softmax(dim=1)[0]
                
                clip_real_prob = clip_probs[0].item()
                clip_ai_prob = clip_probs[1].item()
                clip_edited_prob = clip_probs[2].item()
                clip_fake_prob = clip_ai_prob + clip_edited_prob
                
                # Dynamic Ensemble Weighting
                # If CLIP strongly detects semantic AI anomalies, we reduce CNN authority
                if clip_fake_prob > 0.6:
                    weight_cnn = 0.2
                    weight_clip = 0.8
                else:
                    weight_cnn = 0.5
                    weight_clip = 0.5
                
                final_real_prob = (weight_cnn * real_prob) + (weight_clip * clip_real_prob)
                final_fake_prob = (weight_cnn * fake_prob) + (weight_clip * clip_fake_prob)
            else:
                final_real_prob = real_prob
                final_fake_prob = fake_prob
                
            final_confidence = max(final_real_prob, final_fake_prob)
            is_real = final_real_prob > final_fake_prob
            
            if is_real and final_confidence > 0.55:
                result = "Real"
            elif not is_real:
                # Use CLIP to guide the exact manipulation reason
                if clip_ai_prob > clip_edited_prob and clip_ai_prob > 0.3:
                    result = "AI Generated (Deepfake)"
                elif clip_edited_prob > clip_ai_prob and clip_edited_prob > 0.3:
                    result = "Photoshop Edited"
                else:
                    # Fallback to confidence brackets
                    if final_confidence > 0.85:
                        result = "AI Generated (Deepfake)"
                    elif final_confidence > 0.65:
                        result = "Photoshop Edited"
                    elif final_confidence > 0.5:
                        result = "Filtered / Lightly Modified"
                    else:
                        result = "Uncertain / Possibly Edited"
            else:
                result = "Uncertain / Possibly Edited"
                
            confidence_pct = final_confidence * 100
            
        return jsonify({
            'success': True,
            'result': result,
            'confidence': f"{confidence_pct:.2f}%",
            'real_percentage': f"{final_real_prob * 100:.2f}%",
            'fake_percentage': f"{final_fake_prob * 100:.2f}%",
            'architecture': "Ensemble (EfficientNet+ResNet+MobileNet) + CLIP(ViT-B/32)",
            'image': encoded_img,
            'analysis': compute_analysis(image) if result != "Real" else None
        })

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True, port=5000)
