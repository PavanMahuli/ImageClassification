# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 22:34:20 2020
@author: Krish Naik
"""

import os
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
import torch
import torchvision.transforms as transforms
from PIL import Image

app = Flask(__name__)

# Define the transformation to apply to the images
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load the pre-trained ResNet-50 model
model = torch.hub.load('pytorch/vision', 'resnet50', pretrained=False)
model.fc = torch.nn.Linear(2048, 9)  # Modify the last fully connected layer to match the number of classes

# Load the trained weights
model.load_state_dict(torch.load('classifier_model.pth', map_location=torch.device('cpu')))

model.eval()

# Set the device for inference
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)


def model_predict(img_path):
    # Load and preprocess the image
    img = Image.open(img_path).convert('RGB')
    img = transform(img)
    img = img.unsqueeze(0)  # Add a batch dimension

    # Move the image to the device
    img = img.to(device)

    # Perform the prediction
    with torch.no_grad():
        outputs = model(img)
        _, predicted_class = torch.max(outputs, 1)

    # Convert the predicted class index to a label
    label_mapping = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i']
    predicted_label = label_mapping[predicted_class.item()]
    if predicted_label in ['a', 'c', 'd', 'e', 'f', 'g', 'i']:
        return '5'
    elif predicted_label == 'h':
        return '3'
    else:
        return '1'


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from the POST request
        f = request.files['file']

        # Save the file to the uploads folder
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        preds = model_predict(file_path)
        result = preds
        return result

    return None


if __name__ == '__main__':
    app.run(debug=True)
