import cv2
import torch
import numpy as np
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from PIL import Image
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
from pathlib import Path
import datetime

start_time = datetime.datetime.now()


# === Setup ===
output_dir = Path("entropy_selected_frames")
output_dir.mkdir(exist_ok=True)

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

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Load SAM2
sam2_checkpoint = r"C:\Users\naysh\ML Robocon\image_segmentation\sam2\checkpoints\sam2_hiera_tiny.pt"
model_cfg = "sam2_hiera_t"
sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=device)
mask_generator = SAM2AutomaticMaskGenerator(model=sam2_model)

# Load Grounding DINO
model_id_gdino = "IDEA-Research/grounding-dino-base"
processor_gdino = AutoProcessor.from_pretrained(model_id_gdino, use_auth_token=True)
model_gdino = AutoModelForZeroShotObjectDetection.from_pretrained(model_id_gdino, use_auth_token=True).to(device)
prompt = "Something that doesn't belong on martian surface or terrain."

# === Webcam capture ===
cap = cv2.VideoCapture(0)
fps = 30
duration = 10
frame_limit = int(fps * duration)

frames = []
entropies = []
motion_scores = []

prev_gray = None

print(f"Capturing for {duration} seconds...")

while len(frames) < frame_limit:
    ret, frame = cap.read()
    if not ret:
        break

    # frames.append(frame)

    # # === Entropy calculation ===
    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    # hist_norm = hist / hist.sum()
    # entropy = -np.sum(hist_norm * np.log2(hist_norm + 1e-8))
    # entropies.append(entropy)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if prev_gray is not None:
        # Compute absolute difference between current and previous frame
        diff = cv2.absdiff(gray, prev_gray)
        motion_score = np.sum(diff)
        motion_scores.append(motion_score)
    else:
        motion_scores.append(0)  # First frame, no motion score

    prev_gray = gray
    frames.append(frame)

cap.release()
cv2.destroyAllWindows()
print(f"Captured {len(frames)} frames.")

# === Select top-5 frames ===
top_n = 5
top_indices = np.argsort(motion_scores)[-top_n:]

# for rank, idx in enumerate(sorted(top_indices)):
#     frame = frames[idx]
#     print(f"[Frame {idx}] Entropy: {entropies[idx]:.2f}")

for i, idx in enumerate(sorted(top_indices)):
    frame = frames[idx]
    print(f"[Frame {idx}] Motion: {motion_scores[idx]:.0f}")
    
    # === SAM2 segmentation ===
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    masks = mask_generator.generate(frame_rgb)

    overlay = np.zeros_like(frame, dtype=np.uint8)

    for mask_idx, mask_dict in enumerate(masks):
        mask = mask_dict["segmentation"].astype(np.uint8)

        if np.sum(mask) < 5000:
            continue

        color = COLORS[mask_idx % len(COLORS)]
        overlay[mask == 1] = color

        segmented = cv2.bitwise_and(frame, frame, mask=mask)

        y_indices, x_indices = np.where(mask == 1)
        if len(y_indices) == 0 or len(x_indices) == 0:
            continue
        x_min, x_max = np.min(x_indices), np.max(x_indices)
        y_min, y_max = np.min(y_indices), np.max(y_indices)
        cropped = segmented[y_min:y_max+1, x_min:x_max+1]

        if cropped.shape[0] < 10 or cropped.shape[1] < 10:
            continue

        # === Grounding DINO ===
        pil_crop = Image.fromarray(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))
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

    result = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)

    # === Save final result ===
    # out_path = output_dir / f"frame_{idx}_entropy_{entropies[idx]:.2f}.jpg"
    # cv2.imwrite(str(out_path), result)
    # print(f"[Saved] {out_path}")
    
    out_path = output_dir / f"frame_{idx}_motion_{motion_scores[idx]:.0f}.jpg"
    cv2.imwrite(str(out_path), result)
    print(f"[Saved] {out_path}")

endtime = datetime.datetime.now()
print(endtime-start_time)