# Required Libraries
import torch
import numpy as np
import cv2 as CompVision
from PIL import Image
from transformers import AutoImageProcessor, AutoModel
import torchvision.transforms as T
from sklearn.metrics.pairwise import cosine_similarity

# ------------------------------
# Step 1: Load DINOv2 Model
# ------------------------------
processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
model = AutoModel.from_pretrained("facebook/dinov2-base")
model.eval()
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# ------------------------------
# Step 2: Preprocessing Transform
# ------------------------------
transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def get_dino_embedding(crop_bgr):
    crop_rgb = crop_bgr[..., ::-1]  # Convert BGR to RGB
    crop_pil = Image.fromarray(crop_rgb)
    pixel_tensor = transform(crop_pil).unsqueeze(0).to(device)
    with torch.no_grad():
        features = model(pixel_tensor).last_hidden_state[:, 0, :]  # CLS token
    return features.squeeze().cpu().numpy()


# for mask in prd_masks:
#     binary_mask = (torch.sigmoid(mask[0]).detach().cpu().numpy() > 0.5).astype(np.uint8)
#     x, y, w, h = cv2.boundingRect(binary_mask)
#     if w < 10 or h < 10:
#         continue  # Skip tiny segments
#     crop = frame[y:y+h, x:x+w]

#     # Get DINO embedding and check if weird
#     embedding = get_dino_embedding(crop) 