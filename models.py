import streamlit as st
from PIL import Image
from torchvision import models, transforms
import torch
# from torchvision.models import ResNet50_Weights

def predict(image):
    model = get_model()
    
    transform = transforms.Compose([
       transforms.Resize(178),
       transforms.CenterCrop((178, 178)),
       transforms.Resize((128, 128)),
       transforms.ToTensor(), # NOTE: transforms.ToTensor() already divides pixels by 255. internally
       # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    img = Image.open(image)
    input = torch.unsqueeze(transform(img), 0)

    target_labels = ['Female', 'Male']

    with torch.no_grad():
        model.eval()
        output = model(input)
        index = output.data.cpu().numpy().argmax()
        classes = target_labels[index]
        return classes

@st.cache_resource
def get_model():
    model = models.resnet50() # model = models.resnet50(weights=ResNet50_Weights.DEFAULT)
    num_in_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_in_ftrs, 2) # 2 classes output
    
    weights = torch.load('p1_resnet_best_point.pth', map_location=torch.device('cpu'))['model']

    if weights is not None:
        model.load_state_dict(weights)
    
    return model
