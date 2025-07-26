import streamlit as st
import torch
import torchvision
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import io
import google.generativeai as genai
import time

# Configure the Google Generative AI
genai.configure(api_key="AIzaSyBWdaGV6O-nB3PMFA589E75ranScd9WulU")

# Load the trained model
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights=None)
num_classes = 21  # 20 classes + 1 background
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)
in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
hidden_layer = 256
model.roi_heads.mask_predictor = torchvision.models.detection.mask_rcnn.MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes)
model.load_state_dict(torch.load('mask_rcnn_skin_disease.pth', map_location=device))
model.to(device)
model.eval()

# Define transformations
def get_transform(train):
    transforms = []
    transforms.append(torchvision.transforms.ToTensor())
    return torchvision.transforms.Compose(transforms)

# Function to make predictions
def predict(image):
    transform = get_transform(train=False)
    image = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        prediction = model(image)
    
    return prediction

# Dictionary to map label IDs to disease names
label_to_disease = {
    1: 'Acne-and-Rosacea',
    2: 'Athlete-s-foot',
    3: 'Chickenmonkey pox',
    4: 'Cold-Sores',
    5: 'Contact-Dermatitis',
    6: 'Eczema',
    7: 'Hives',
    8: 'Keratosis pilaris',
    9: 'Lupus',
    10: 'Moles',
    11: 'Psoriasis',
    12: 'Ringworm',
    13: 'Shingles',
    14: 'Skin-cancer-Basal-cell-carcinoma-',
    15: 'Skin-cancer-Melanoma-',
    16: 'Skin-cancer-Squamous-cell-carcinoma-',
    17: 'Vitiligo',
    18: 'Warts',
    19: 'cyst',
    20: 'nail-fungus'
}

# Function to visualize predictions
def visualize_prediction(image, prediction):
    plt.figure(figsize=(12, 12))
    plt.imshow(image)
    
    pred_boxes = prediction[0]['boxes'].cpu().numpy()
    pred_labels = prediction[0]['labels'].cpu().numpy()
    pred_scores = prediction[0]['scores'].cpu().numpy()
    pred_masks = prediction[0]['masks'].cpu().numpy()
    
    for i in range(len(pred_boxes)):
        if pred_scores[i] > 0.5:  # Only display predictions with a confidence score above 0.5
            box = pred_boxes[i]
            mask = pred_masks[i, 0]
            disease_name = label_to_disease.get(pred_labels[i], 'Unknown')
            plt.gca().add_patch(plt.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1], 
                                              fill=False, edgecolor='red', linewidth=2))
            plt.text(box[0], box[1], f'{disease_name}: {pred_labels[i]}', color='red', fontsize=12)
            plt.imshow(mask, alpha=0.5, cmap='jet')
    
    plt.axis('off')
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    return buf


# Streamlit app
st.title("Skin Disease Detection using Mask R-CNN")
uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded Image', use_column_width=True)
    st.write("")
    st.write("Classifying...")
    
    prediction = predict(image)
    buf = visualize_prediction(image, prediction)
    st.image(buf, caption='Predicted Image', use_column_width=True)
    
    # Print class names and explanations
    pred_labels = prediction[0]['labels'].cpu().numpy()
    pred_scores = prediction[0]['scores'].cpu().numpy()
    st.write("Detected Classes:")
    
    processed_diseases = set()
    
    for i in range(len(pred_labels)):
        if pred_scores[i] > 0.5:
            disease_name = label_to_disease.get(pred_labels[i], 'Unknown')
            if disease_name not in processed_diseases:
                st.write(f"{disease_name}: {pred_scores[i]:.2f}")
                processed_diseases.add(disease_name)
                
                # Generate explanation using Google Generative AI
                model = genai.GenerativeModel("gemini-1.5-flash")
                response = model.generate_content(f"Explain about {disease_name} in a paragraph and give start with a soothing tone to the paitent and suggest a remedy in a seperate paragraph.")
                
                # Gradually display the generated text
                explanation_placeholder = st.empty()
                explanation_text = response.text
                displayed_text = ""
                for char in explanation_text:
                    displayed_text += char
                    explanation_placeholder.markdown(f"<div style='word-wrap: break-word;'>{displayed_text}</div>", unsafe_allow_html=True)
                    time.sleep(0.01)