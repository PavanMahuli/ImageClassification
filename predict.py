import torch
import torchvision.transforms as transforms
from PIL import Image
import torchvision.models as models
import torch.nn as nn

# Define the transformation to apply to the images
# Load the pre-trained ResNet-50 model
model = models.resnet50(pretrained=False)
model.fc = nn.Linear(2048, 9)  # Modify the last fully connected layer to match the number of classes

# Load the trained weights
model.load_state_dict(torch.load('classifier_model.pth'))
model.eval()

# Set the device for inference
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)


# Define the transformation to apply to the test image
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Rescale the image to 224x224
    transforms.ToTensor(),           # Convert the image to a tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize the image
])

# Load and preprocess the test image
test_image_path = '/content/drive/MyDrive/test/lake_person/man-bikes-along-path-to-lake-in-the-mountains-as-storm-approaches.jpg'
test_image = Image.open(test_image_path).convert('RGB')
test_image = transform(test_image)
test_image = test_image.unsqueeze(0)  # Add a batch dimension

# Move the test image to the device
test_image = test_image.to(device)

# Perform the prediction
with torch.no_grad():
    outputs = model(test_image)
    _, predicted_class = torch.max(outputs, 1)

# Convert the predicted class index to a label
label_mapping = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i']
predicted_label = label_mapping[predicted_class.item()]
if predicted_label in ['a', 'c', 'd', 'e', 'f', 'g', 'i']:
  print('5')
# Print the predicted label
#print("Predicted label:", predicted_label)
elif predicted_label=='h':
  print('3')
else:
  print('1')