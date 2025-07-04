import cv2
import torch
import numpy as np
from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from get_embeddings import get_dino_embedding
import datetime
import os

# === Create output directory if not exists ===
os.makedirs("normal_data/images", exist_ok=True)

# === Load SAM2 model ===
sam2_checkpoint = r"C:\Users\naysh\ML Robocon\image_segmentation\sam2\checkpoints\sam2_hiera_tiny.pt"
model_cfg = "sam2_hiera_t"  # this will map to config internally in build_sam2
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=device)

# === Initialize Automatic Mask Generator ===
mask_generator = SAM2AutomaticMaskGenerator(model=sam2_model)

# === Load Image ===
image_path = r"C:\Users\naysh\ML Robocon\image_segmentation\dinov2\test_images\farm1_test.jpg"
image_bgr = cv2.imread(image_path)
image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

# === Generate Masks ===
masks = mask_generator.generate(image_rgb)

# === Process Masks and Save ===
normal_embeddings = []

for idx, mask_dict in enumerate(masks):
    mask = mask_dict["segmentation"]  # binary mask: H x W

    # Apply mask to image
    segmented_object = cv2.bitwise_and(image_rgb, image_rgb, mask=mask.astype(np.uint8))

    # Save segmented image
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    seg_img_path = f"normal_data/images/segmented_{timestamp}_{idx}.jpg"
    cv2.imwrite(seg_img_path, cv2.cvtColor(segmented_object, cv2.COLOR_RGB2BGR))
    print(f"Saved segmented image to {seg_img_path}")

    # Get DINOv2 embedding
    embedding = get_dino_embedding(segmented_object)
    normal_embeddings.append(embedding)

# === Save all embeddings as .npy ===
embedding_path = "normal_data/normal_embeddings.npy"
np.save(embedding_path, np.array(normal_embeddings))

normal_embeddings = np.load("normal_embeddings.npy")
assert normal_embeddings.ndim == 2  # (N, D)

print(f"Saved {len(normal_embeddings)} embeddings to {embedding_path}")
