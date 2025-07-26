import torch
import torchvision
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

# Load the trained model
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights=None)
num_classes = 21  # 20 classes + 1 background
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)
in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
hidden_layer = 256
model.roi_heads.mask_predictor = torchvision.models.detection.mask_rcnn.MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes)
model.load_state_dict(torch.load('mask_rcnn_skin_disease.pth'))
model.to(device)
model.eval()

# Define transformations
def get_transform(train):
    transforms = []
    transforms.append(torchvision.transforms.ToTensor())
    return torchvision.transforms.Compose(transforms)

# Function to make predictions
def predict(image_path):
    image = Image.open(image_path).convert('RGB')
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
def visualize_prediction(image_path, prediction):
    image = Image.open(image_path).convert('RGB')
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
    plt.show()

# Example usage
image_path = r'c:\Users\Rajkumar\Downloads\ch851.fig1_.jpg'  # Use raw string
# or use forward slashes
# image_path = 'c:/Users/CLASSIC/Downloads/types-stages-of-toenail-fungus.jpg'
prediction = predict(image_path)
visualize_prediction(image_path, prediction)