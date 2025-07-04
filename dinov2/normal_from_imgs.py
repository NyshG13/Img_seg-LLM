import cv2
import torch
import numpy as np
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from hydra import initialize, compose
from hydra.core.global_hydra import GlobalHydra
from get_embeddings import get_dino_embedding
from scipy.spatial.distance import cosine
import datetime

import os
normal_embeddings = []

os.makedirs("normal_data", exist_ok=True)

# Load the SAM2 model
sam2_checkpoint = r"C:\Users\naysh\ML Robocon\image_segmentation\sam2\checkpoints\sam2_hiera_tiny.pt"
# model_cfg = r"C:\Users\naysh\ML Robocon\image_segmentation\sam2\sam2\sam2_hiera_t.yaml"

model_cfg = "sam2_hiera_t"
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=device)
predictor = SAM2ImagePredictor(sam2_model)

# Load image
image_path = r"C:\Users\naysh\ML Robocon\image_segmentation\dinov2\test_images\farm1_test.jpg"
image_bgr = cv2.imread(image_path)
image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

# Predict
predictor.set_image(image_rgb)

# Dummy center point as prompt (you can change this to other coordinates)
h, w, _ = image_bgr.shape
input_points = np.array([[[w // 2, h // 2]]])
input_labels = np.ones([1, 1])

# Prepare prompts
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

# Apply mask to image (black background)
segmented_object = cv2.bitwise_and(image_rgb, image_rgb, mask=mask)

# Save segmented image
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
seg_img_path = f"normal_data/images/segmented_{timestamp}.jpg"
cv2.imwrite(seg_img_path, cv2.cvtColor(segmented_object, cv2.COLOR_RGB2BGR))

# Get DINOv2 embedding
embedding = get_dino_embedding(segmented_object)  # Must return numpy array

# Save embedding
embedding_path = f"normal_data/normal_embeddings.npy"
# np.save(embedding_path, embedding)
np.save(embedding_path, np.array(normal_embeddings))


print(f"Saved segmented image to {seg_img_path}")
print(f"Saved embedding to {embedding_path}")

# Overlay mask
# overlay = image_bgr.copy()
# overlay[mask == 1] = (0, 255, 0)  # Green
# result = cv2.addWeighted(image_bgr, 0.7, overlay, 0.3, 0)

# # Show and/or save result
# cv2.imshow("Segmented Image", result)
# cv2.waitKey(0)


cv2.destroyAllWindows()
# Optional: Save the result
# cv2.imwrite(r"C:\Users\naysh\ML Robocon\image_segmentation\dinov2\img_output\segmented_image.jpg", result)
