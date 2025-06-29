import cv2 as CompVision
import torch
import numpy as np
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from hydra import initialize, compose
from hydra.core.global_hydra import GlobalHydra
from get_embeddings import get_dino_embedding
from scipy.spatial.distance import cosine


import os
normal_embeddings = []

# normal_embeddings = np.load("normal_data/normal_embeddings.npy", allow_pickle=True)

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

# OpenCV webcam capture
# cap = cv2.VideoCapture(0)
cap = CompVision.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame_rgb = CompVision.cvtColor(frame, CompVision.COLOR_BGR2RGB)
    predictor.set_image(frame_rgb)
    
    # Generate dummy points (modify this part for actual interactive points)
    h, w, _ = frame.shape
    input_points = np.array([[[w//2, h//2]]])
    input_labels = np.ones([1, 1])
    
    mask_input, unnorm_coords, labels, unnorm_box = predictor._prep_prompts(
        input_points, input_labels, box=None, mask_logits=None, normalize_coords=True
    )
    sparse_embeddings, dense_embeddings = predictor.model.sam_prompt_encoder(
        points=(unnorm_coords, labels), boxes=None, masks=None
    )
    
    # Generate mask
    low_res_masks, prd_scores, _, _ = predictor.model.sam_mask_decoder(
        image_embeddings=predictor._features["image_embed"][-1].unsqueeze(0),
        image_pe=predictor.model.sam_prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=sparse_embeddings,
        dense_prompt_embeddings=dense_embeddings,
        multimask_output=True,
        repeat_image=False,
        high_res_features=[feat_level[-1].unsqueeze(0) for feat_level in predictor._features["high_res_feats"]]
    )
    prd_masks = predictor._transforms.postprocess_masks(low_res_masks, predictor._orig_hw[-1])
    mask = (torch.sigmoid(prd_masks[0, 0]).detach().cpu().numpy() > 0.5).astype(np.uint8)

    
    # Overlay mask
    overlay = frame.copy()
    overlay[mask == 1] = (0, 255, 0)  # Green mask
    result = CompVision.addWeighted(frame, 0.7, overlay, 0.3, 0)
    
    CompVision.imshow("Segmented Output", result)

        # If valid mask
    y_indices, x_indices = np.where(mask == 1)
    if len(y_indices) == 0 or len(x_indices) == 0:
        continue

    x_min, x_max = np.min(x_indices), np.max(x_indices)
    y_min, y_max = np.min(y_indices), np.max(y_indices)
    cropped_object = frame[y_min:y_max + 1, x_min:x_max + 1]

    if cropped_object.shape[0] < 10 or cropped_object.shape[1] < 10:
        continue

    emb = get_dino_embedding(cropped_object)

    key = CompVision.waitKey(1) & 0xFF

    if key == ord('n'):
        print("Saving this object as NORMAL")

        y_indices, x_indices = np.where(mask == 1)
        if len(y_indices) == 0 or len(x_indices) == 0:
            print("Empty mask. Skipping.")
            continue

        x_min, x_max = np.min(x_indices), np.max(x_indices)
        y_min, y_max = np.min(y_indices), np.max(y_indices)
        cropped_object = frame[y_min:y_max + 1, x_min:x_max + 1]

        if cropped_object.shape[0] < 10 or cropped_object.shape[1] < 10:
            print("Object too small to save. Skipping.")
            continue

        emb = get_dino_embedding(cropped_object)
        normal_embeddings.append(emb)

        CompVision.imwrite(f"normal_data/normal_{len(normal_embeddings)}.jpg", cropped_object)
        print(f"Saved embedding #{len(normal_embeddings)}")


    elif key == ord('s'):
        np.save("normal_data/normal_embeddings.npy", np.array(normal_embeddings))
        print(f"Saved {len(normal_embeddings)} embeddings to normal_embeddings.npy")

    elif key == ord('q'):
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
