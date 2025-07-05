import cv2 as CompVision
import torch
import numpy as np
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
import cv2
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
import datetime 

import os

os.makedirs("weird_objects", exist_ok=True)

COLORS = [
    (255, 0, 0),     # Blue
    (0, 255, 0),     # Green
    (0, 0, 255),     # Red
    (255, 255, 0),   # Cyan
    (255, 0, 255),   # Magenta
    (0, 255, 255),   # Yellow
    (128, 0, 128),   # Purple
    (0, 128, 128),   # Teal
    (128, 128, 0),   # Olive
    (192, 192, 192)  # Silver
]

# Load the SAM2 model
sam2_checkpoint = r"C:\Users\naysh\ML Robocon\image_segmentation\sam2\checkpoints\sam2_hiera_tiny.pt"

model_cfg = "sam2_hiera_t"

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=device)
predictor = SAM2ImagePredictor(sam2_model)

mask_generator = SAM2AutomaticMaskGenerator(model=sam2_model)

model_id_gdino = "IDEA-Research/grounding-dino-base"
device = "cuda" if torch.cuda.is_available() else "cpu"

processor_gdino = AutoProcessor.from_pretrained(model_id_gdino, use_auth_token=True)
model_gdino = AutoModelForZeroShotObjectDetection.from_pretrained(model_id_gdino, use_auth_token=True).to(device)

prompt = "Something that doesn't belong on martian surface or terrain."
#"An item that looks strange or out of place on Mars."
# "Something that doesn't fit in a natural Mars environment."


# OpenCV webcam capture
cap = CompVision.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame_rgb = CompVision.cvtColor(frame, CompVision.COLOR_BGR2RGB)

    try:
        # Generate masks for current frame
        masks = mask_generator.generate(frame_rgb)

        # Prepare mask overlay
        overlay = np.zeros_like(frame, dtype=np.uint8)

        for mask_idx, mask_dict in enumerate(masks):
            mask = mask_dict["segmentation"].astype(np.uint8)

            # Optional area filter
            if np.sum(mask) < 5000:
                continue

            # Add green overlay
            color = COLORS[mask_idx % len(COLORS)]  # Cycle through COLORS
            overlay[mask == 1] = color

            # Apply mask to original image to extract the object
            segmented_object = CompVision.bitwise_and(frame, frame, mask=mask)

            # Crop the mask region (tight bounding box)
            y_indices, x_indices = np.where(mask == 1)
            if len(y_indices) == 0 or len(x_indices) == 0:
                continue
            x_min, x_max = np.min(x_indices), np.max(x_indices)
            y_min, y_max = np.min(y_indices), np.max(y_indices)
            cropped_object = segmented_object[y_min:y_max + 1, x_min:x_max + 1]

            # Skip if too small
            if cropped_object.shape[0] < 10 or cropped_object.shape[1] < 10:
                continue

            #apply grounding dino
            pil_crop = Image.fromarray(CompVision.cvtColor(cropped_object, CompVision.COLOR_BGR2RGB))


            inputs = processor_gdino(images=pil_crop, text=prompt, return_tensors="pt").to(device)

            with torch.no_grad():
                outputs = model_gdino(**inputs)

            # Post-process
            results = processor_gdino.post_process_grounded_object_detection(
                outputs,
                inputs.input_ids,
                box_threshold=0.4,        # tuneable
                text_threshold=0.3,       # tuneable
                target_sizes=[pil_crop.size[::-1]]  # (height, width)
            )

            if len(results[0]["boxes"]) > 0:
            # Construct filename with timestamp
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                filename = f"weird_objects/weird_{timestamp}.jpg"

                # Save the cropped object (OpenCV format)
                CompVision.imwrite(filename, cropped_object)

                print(f"[INFO] Unusual object saved: {filename}")

        result = CompVision.addWeighted(frame, 0.7, overlay, 0.3, 0)
    except Exception as e:
        print("Mask generation failed:", e)
        result = frame

    CompVision.imshow("SAM2 Live Segmentation", result)

    key = CompVision.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cap.release()
CompVision.destroyAllWindows()
