import os
import io
import base64
import numpy as np
from PIL import Image
from flask import Flask, render_template, request, jsonify
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
import cv2
import timm

app = Flask(__name__)
UPLOAD_FOLDER = 'static'
MODEL_DIR = 'model'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

classes = ['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative']

# Model-specific transforms
transforms_dict = {
    'resnet50': transforms.Compose([
        transforms.Resize((224, 224), interpolation=Image.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]),
    'efficientnet': transforms.Compose([
        transforms.Resize((224, 224), interpolation=Image.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]),
    'mobilenetv4': transforms.Compose([
        transforms.Resize((384, 384), interpolation=Image.BILINEAR),  # MobileNetV4-Conv-Large uses 384x384
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
}

def load_model(arch, path):
    if not os.path.exists(path):
        print(f"Model file not found at {path}")
        return None, None

    try:
        if arch == 'resnet50':
            model = models.resnet50(weights=None)
            model.fc = nn.Linear(model.fc.in_features, 5)
            final_conv_layer = model.layer4[-1].conv3
        elif arch == 'efficientnet':
            model = models.efficientnet_b0(weights=None)
            model.classifier[1] = nn.Linear(model.classifier[1].in_features, 5)
            final_conv_layer = model.features[-1][0]
        elif arch == 'vgg16':
            model = models.vgg16(weights=None)
            model.classifier[6] = nn.Linear(model.classifier[6].in_features, 5)
            final_conv_layer = model.features[-1]  # Last conv layer (MaxPool2d follows in VGG16)
        elif arch == 'mobilenetv4':
            model = timm.create_model('mobilenetv4_conv_large.e600_r384_in1k', pretrained=False, num_classes=5)
            final_conv_layer = model.conv_head  # Final conv layer before classifier
        else:
            raise ValueError(f"Unsupported architecture: {arch}")

        model.load_state_dict(torch.load(path, map_location=device))
        model.to(device).eval()
        return model, final_conv_layer
    except Exception as e:
        print(f"Error loading weights for {arch}: {e}")
        return None, None

# Load models
models_dict = {}
try:
    models_dict['resnet50'] = load_model('resnet50', os.path.join(MODEL_DIR, 'model1.pth'))
    models_dict['efficientnet'] = load_model('efficientnet', os.path.join(MODEL_DIR, 'model2.pth'))
    models_dict['mobilenetv4'] = load_model('mobilenetv4', os.path.join(MODEL_DIR, 'model4.pth'))
except Exception as e:
    print(f"Error loading models: {e}")

# Filter out None models
models_dict = {k: v for k, v in models_dict.items() if v[0] is not None}

if not models_dict:
    raise RuntimeError("No valid models loaded. Please check model files.")

def generate_gradcam(model, final_conv, img_tensor, target_class=None):
    model.eval()
    gradients = []
    activations = []

    def save_gradient(grad):
        gradients.append(grad)

    def forward_hook(module, input, output):
        activations.append(output)
        output.register_hook(save_gradient)

    handle = final_conv.register_forward_hook(forward_hook)

    try:
        output = model(img_tensor)
        if output.dim() != 2 or output.shape[1] != 5:
            raise RuntimeError(f"Unexpected model output shape: {output.shape}")
        
        class_idx = output.argmax(dim=1).item() if target_class is None else target_class
        loss = output[0, class_idx]

        model.zero_grad()
        loss.backward()
    finally:
        handle.remove()

    if not gradients or not activations:
        raise RuntimeError("Failed to capture gradients or activations")

    gradients_ = gradients[0].detach().cpu().numpy()[0]
    activations_ = activations[0].detach().cpu().numpy()[0]

    weights = np.mean(gradients_, axis=(1, 2))
    cam = np.zeros(activations_.shape[1:], dtype=np.float32)
    for i, w in enumerate(weights):
        cam += w * activations_[i]
    cam = np.maximum(cam, 0)
    if cam.max() != 0:
        cam = cam / cam.max()

    # Resize to match input image size
    input_size = img_tensor.shape[-2:]  # (height, width)
    cam = cv2.resize(cam, (input_size[1], input_size[0]))
    cam = (cam * 255).astype(np.uint8)
    heatmap = cv2.applyColorMap(cam, cv2.COLORMAP_JET)
    return heatmap

def overlay_heatmap(original_img, heatmap, size=(224, 224)):
    img = np.array(original_img.resize(size, resample=Image.BILINEAR))
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    overlay = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)
    overlay = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
    return overlay

def encode_img(img_array):
    _, buffer = cv2.imencode('.png', img_array)
    b64 = base64.b64encode(buffer).decode('utf-8')
    return b64

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    try:
        img = Image.open(file).convert('RGB')
    except Exception:
        return jsonify({'error': 'Invalid image file'}), 400

    models_list = [
        ("ResNet50", models_dict.get('resnet50', (None, None)), transforms_dict['resnet50'], (224, 224)),
        ("EfficientNet", models_dict.get('efficientnet', (None, None)), transforms_dict['efficientnet'], (224, 224)),
        ("MobileNetV4", models_dict.get('mobilenetv4', (None, None)), transforms_dict['mobilenetv4'], (384, 384))
    ]

    results = []
    predictions = []

    for name, (model, final_conv), transform, size in models_list:
        if model is None or final_conv is None:
            results.append({
                'model': name,
                'prediction': 'Error',
                'confidence': '0.00%',
                'explanation': f'Model {name} not loaded due to missing or invalid weights.',
                'heatmap': None
            })
            continue

        try:
            img_tensor = transform(img).unsqueeze(0).to(device)
            with torch.no_grad():
                output = model(img_tensor)
                probs = torch.softmax(output, dim=1)[0]
                conf, pred = torch.max(probs, dim=0)

            heatmap = generate_gradcam(model, final_conv, img_tensor, pred.item())
            overlay = overlay_heatmap(img, heatmap, size=size)
            b64_img = encode_img(overlay)

            results.append({
                'model': name,
                'prediction': classes[pred.item()],
                'confidence': f'{conf.item()*100:.2f}%',
                'explanation': f'{classes[pred.item()]} detected with {conf.item()*100:.2f}% confidence.',
                'heatmap': b64_img
            })

            predictions.append(pred.item())
        except Exception as e:
            print(f"Error processing model {name}: {e}")
            results.append({
                'model': name,
                'prediction': 'Error',
                'confidence': '0.00%',
                'explanation': f'Failed to process: {str(e)}',
                'heatmap': None
            })

    if not any(r['prediction'] != 'Error' for r in results):
        return jsonify({'error': 'All models failed to process'}), 500

    # Majority voting for final prediction
    valid_predictions = [p for p in predictions if p is not None]
    if valid_predictions:
        final_pred = max(set(valid_predictions), key=valid_predictions.count)
        valid_results = [r for r in results if classes.index(r['prediction']) == final_pred and r['prediction'] != 'Error']
        avg_conf = np.mean([float(r['confidence'].rstrip('%')) for r in valid_results]) if valid_results else 0.0
        final_expl = f'Based on majority voting, the condition is {classes[final_pred]}.'
    else:
        final_pred = None
        avg_conf = 0.0
        final_expl = 'No valid predictions available.'

    return jsonify({
        'results': results,
        'final_prediction': classes[final_pred] if final_pred is not None else 'None',
        'final_confidence': f'{avg_conf:.2f}%',
        'final_explanation': final_expl
    })

if __name__ == '__main__':
    app.run(debug=True)