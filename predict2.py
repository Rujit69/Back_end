import torch
from torchvision import transforms, models
from PIL import Image
import torch.nn.functional as F
import torch.nn as nn
from flask import Flask, request, jsonify
from flask_cors import CORS
import io  # For handling byte streams

app = Flask(__name__)
CORS(app)  # Enable CORS for React frontend

# Define transformations and load model (same as before)
data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

checkpoint_path = 'best_model.pth'
class_names = ['fake', 'real']

def load_model():
    model = models.densenet121(weights='IMAGENET1K_V1')
    model.classifier = nn.Sequential(
        nn.Linear(1024, 512),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(512, 2)
    )
    # Ensure model loads on CPU regardless of the system's GPU availability
    device = torch.device('cpu')
    model = model.to(device)
    
    # Load the model checkpoint and map to CPU
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model, device

model, device = load_model()

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'Empty filename'}), 400

    try:
        # Read image bytes directly without saving to disk
        image_bytes = file.read()
        image = Image.open(io.BytesIO(image_bytes))

        # Convert RGBA to RGB if needed
        if image.mode == 'RGBA':
            image = image.convert('RGB')

        # Apply transformations and predict
        image = data_transforms(image).unsqueeze(0).to(device)
        with torch.no_grad():
            outputs = model(image)
            probabilities = F.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
            confidence = confidence.item()

        return jsonify({
            'prediction': class_names[predicted.item()],
            'confidence': f"{confidence * 100:.2f}%"
        })

    except Exception as e:
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
