import cv2 as CompVision
import torch
import numpy as np
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from hydra import initialize, compose
from hydra.core.global_hydra import GlobalHydra
from get_embeddings import get_dino_embedding
from scipy.spatial.distance import cosine
from anamoly_checker import is_anomalous
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
import cv2
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
import datetime 

import os
# normal_embeddings = []

normal_embeddings = np.load("normal_data/normal_embeddings.npy", allow_pickle=True)
if normal_embeddings.shape == (1,) and isinstance(normal_embeddings[0], (list, np.ndarray)):
    normal_embeddings = normal_embeddings[0]

os.makedirs("normal_data", exist_ok=True)

# Load the SAM2 model
sam2_checkpoint = r"C:\Users\naysh\ML Robocon\image_segmentation\sam2\checkpoints\sam2_hiera_tiny.pt"
# model_cfg = r"C:\Users\naysh\ML Robocon\image_segmentation\sam2\sam2\sam2_hiera_t.yaml"

# if not GlobalHydra.instance().is_initialized():
#     GlobalHydra.instance().clear()

# with initialize(config_path=r"sam2\sam2\sam2_hiera_t.yaml", version_base=None):
#     model_cfg = compose(config_name="sam2_hiera_t")

model_cfg = "sam2_hiera_t"

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=device)
predictor = SAM2ImagePredictor(sam2_model)

mask_generator = SAM2AutomaticMaskGenerator(model=sam2_model)

# OpenCV webcam capture
# cap = cv2.VideoCapture(0)
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
            overlay[mask == 1] = (0, 255, 0)

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
            model_id = "IDEA-Research/grounding-dino-base"
            device = "cuda" if torch.cuda.is_available() else "cpu"

            processor = AutoProcessor.from_pretrained(model_id, use_auth_token=True)
            model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id, use_auth_token=True).to(device)

            prompt = "an odd item. Something that doesn't belong."

            pil_crop = Image.fromarray(CompVision.cvtColor(cropped_object, CompVision.COLOR_BGR2RGB))

            # Grounding DINO input
            inputs = processor(images=pil_crop, text=prompt, return_tensors="pt").to(device)

            with torch.no_grad():
                outputs = model(**inputs)

            # Post-process
            results = processor.post_process_grounded_object_detection(
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

            # # Get DINOv2 embedding
            # emb = get_dino_embedding(cropped_object).squeeze()
            # if emb.ndim != 1:
            #     emb = emb.reshape(-1)

            # # Check if it's anomalous
            # if is_anomalous(emb, normal_embeddings):
            #     os.makedirs("weird_objects", exist_ok=True)
            #     idx = len(os.listdir("weird_objects")) // 2
            #     path = f"weird_objects/weird_{idx+1}.jpg"
            #     CompVision.imwrite(path, cropped_object)
            #     np.save(f"weird_objects/weird_{idx+1}.npy", emb)
            #     print(f"Weird object saved: {path}")
            # else:
            #     print("Object is normal")

        # Blend original frame with overlay
        result = CompVision.addWeighted(frame, 0.7, overlay, 0.3, 0)
    except Exception as e:
        print("Mask generation failed:", e)
        result = frame

    CompVision.imshow("SAM2 Live Segmentation", result)
    
    # CompVision.imshow("Segmented Output", result)

    #     # If valid mask
    # y_indices, x_indices = np.where(mask == 1)
    # if len(y_indices) == 0 or len(x_indices) == 0:
    #     continue

    # x_min, x_max = np.min(x_indices), np.max(x_indices)
    # y_min, y_max = np.min(y_indices), np.max(y_indices)
    # cropped_object = frame[y_min:y_max + 1, x_min:x_max + 1]

    # if cropped_object.shape[0] < 10 or cropped_object.shape[1] < 10:
    #     continue

    # emb = get_dino_embedding(cropped_object)
    # emb = emb.squeeze()

    # if emb.ndim != 1:
    #     emb = emb.reshape(-1)

    # if is_anomalous(emb, normal_embeddings):
    #     # print("Weird object detected!")

    #     os.makedirs("weird_objects", exist_ok=True)
    #     idx = len(os.listdir("weird_objects")) // 2
    #     path = f"weird_objects/weird_{idx+1}.jpg"
    #     CompVision.imwrite(path, cropped_object)
    #     np.save(f"weird_objects/weird_{idx+1}.npy", emb)
    # else:
    #     print("✅ Object is normal")

    key = CompVision.waitKey(1) & 0xFF
    if key == ord('q'):
        break


    #dinov2
    # # Extract one object from mask
    # y_indices, x_indices = np.where(mask == 1)
    # if len(y_indices) == 0 or len(x_indices) == 0:
    #     continue  # No mask found

    # x_min, x_max = np.min(x_indices), np.max(x_indices)
    # y_min, y_max = np.min(y_indices), np.max(y_indices)

    # # Crop the object
    # cropped_object = frame[y_min:y_max+1, x_min:x_max+1]


cap.release()
CompVision.destroyAllWindows()
