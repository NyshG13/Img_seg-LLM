import cv2 as CompVision
import torch
import numpy as np
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from PIL import Image
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
import datetime 
import os

# === CONFIG ===
IMAGE_PATH = "mars_and_bottle.jpg" 
sam2_checkpoint = "sam2/checkpoints/sam2_hiera_tiny.pt"
model_cfg = "sam2_hiera_t"
model_id_gdino = "IDEA-Research/grounding-dino-base"
prompt = "Something that doesn't belong on martian surface or terrain."

# === Setup ===
start_time = datetime.datetime.now()
os.makedirs("weird_objects", exist_ok=True)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

COLORS = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255),
    (255, 255, 0), (255, 0, 255), (0, 255, 255),
    (128, 0, 128), (0, 128, 128), (128, 128, 0), (192, 192, 192)
]

# === Load models ===
sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=device)
predictor = SAM2ImagePredictor(sam2_model)
mask_generator = SAM2AutomaticMaskGenerator(model=sam2_model)

processor_gdino = AutoProcessor.from_pretrained(model_id_gdino, use_auth_token=True)
model_gdino = AutoModelForZeroShotObjectDetection.from_pretrained(model_id_gdino, use_auth_token=True).to(device)

# === Load image ===
image = CompVision.imread(IMAGE_PATH)
if image is None:
    raise FileNotFoundError(f"Image not found: {IMAGE_PATH}")

frame_rgb = CompVision.cvtColor(image, CompVision.COLOR_BGR2RGB)

try:
    # Generate masks for the image
    masks = mask_generator.generate(frame_rgb)
    overlay = np.zeros_like(image, dtype=np.uint8)

    for mask_idx, mask_dict in enumerate(masks):
        mask = mask_dict["segmentation"].astype(np.uint8)
        if np.sum(mask) < 5000:
            continue

        color = COLORS[mask_idx % len(COLORS)]
        overlay[mask == 1] = color

        segmented_object = CompVision.bitwise_and(image, image, mask=mask)
        y_indices, x_indices = np.where(mask == 1)
        if len(y_indices) == 0 or len(x_indices) == 0:
            continue

        x_min, x_max = np.min(x_indices), np.max(x_indices)
        y_min, y_max = np.min(y_indices), np.max(y_indices)
        cropped_object = segmented_object[y_min:y_max + 1, x_min:x_max + 1]

        if cropped_object.shape[0] < 10 or cropped_object.shape[1] < 10:
            continue

        pil_crop = Image.fromarray(CompVision.cvtColor(cropped_object, CompVision.COLOR_BGR2RGB))
        inputs = processor_gdino(images=pil_crop, text=prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model_gdino(**inputs)

        results = processor_gdino.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            box_threshold=0.4,
            text_threshold=0.3,
            target_sizes=[pil_crop.size[::-1]]
        )

        if len(results[0]["boxes"]) > 0:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            filename = f"weird_objects/weird_{timestamp}.jpg"
            CompVision.imwrite(filename, cropped_object)
            print(f"[INFO] Unusual object saved: {filename}")

except Exception as e:
    print("Processing failed:", e)

endtime = datetime.datetime.now()
print("Total Time:", endtime - start_time)
