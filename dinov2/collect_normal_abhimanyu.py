import cv2 as CompVision
import torch
import numpy as np
from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from get_embeddings import get_dino_embedding
import os

# --- Configuration ---
SAM2_CHECKPOINT = r"C:\Users\naysh\ML Robocon\image_segmentation\sam2\checkpoints\sam2_hiera_tiny.pt"
MODEL_CFG_NAME = "sam2_hiera_t"
NORMAL_DATA_DIR = "normal_data"
NORMAL_EMBEDDINGS_FILE = os.path.join(NORMAL_DATA_DIR, "normal_embeddings.npy")

# --- Initialization ---
os.makedirs(NORMAL_DATA_DIR, exist_ok=True)
normal_embeddings = []

# Load existing embeddings if available
if os.path.exists(NORMAL_EMBEDDINGS_FILE):
    try:
        normal_embeddings = list(np.load(NORMAL_EMBEDDINGS_FILE, allow_pickle=True))
        print(f"Loaded {len(normal_embeddings)} existing normal embeddings.")
    except Exception as e:
        print(f"Could not load normal embeddings: {e}")

# --- Model Loading ---
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

print("Building SAM2 model...")
sam2_model = build_sam2(MODEL_CFG_NAME, SAM2_CHECKPOINT, device=device)

# --- Use the Automatic Mask Generator ---
print("Initializing Automatic Mask Generator...")
mask_generator = SAM2AutomaticMaskGenerator(
    model=sam2_model,
    points_per_side=16,
    pred_iou_thresh=0.86,
    stability_score_thresh=0.92,
    min_mask_region_area=200,
)

# --- OpenCV Webcam Capture ---
cap = CompVision.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

print("\n--- Starting Capture ---")
print("Press 'n' to enter approval mode for the current frame.")
print("Press 's' to save the collected embedding data to a file.")
print("Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture frame.")
        break
    
    frame_rgb = CompVision.cvtColor(frame, CompVision.COLOR_BGR2RGB)
    
    # --- Generate all masks for the current frame ---
    masks = mask_generator.generate(frame_rgb)

    # --- Live Visualization ---
    display_frame = frame.copy()
    if masks:
        # Sort by area so smaller masks are drawn on top
        sorted_masks = sorted(masks, key=lambda x: x['area'], reverse=False)
        overlay = display_frame.copy()
        for mask_data in sorted_masks:
            color = np.random.randint(50, 255, (3,)).tolist()
            overlay[mask_data['segmentation']] = color
        display_frame = CompVision.addWeighted(display_frame, 0.4, overlay, 0.6, 0)
    
    CompVision.imshow("Segmented Output", display_frame)

    # --- User Interaction ---
    key = CompVision.waitKey(1) & 0xFF

    # --- APPROVAL MODE ---
    if key == ord('n'):
        if not masks:
            print("[INFO] No masks detected in the current frame to approve.")
            continue

        print(f"\n--- ENTERING APPROVAL MODE for {len(masks)} masks ---")
        
        # Sort masks by a consistent property, e.g., size, for a predictable order
        masks_to_approve = sorted(masks, key=lambda x: x['area'], reverse=True)
        
        for i, mask_data in enumerate(masks_to_approve):
            # Create a clean frame for highlighting the current mask
            highlight_frame = frame.copy()
            
            # Draw the specific mask being approved in a bright color
            current_mask = mask_data['segmentation']
            highlight_frame[current_mask] = (0, 255, 255) # Bright Yellow

            # Add instructional text
            prompt_text = f"Mask {i+1}/{len(masks_to_approve)} | Approve? (y/n) | Esc to Cancel"
            CompVision.putText(highlight_frame, prompt_text, (10, 30), 
                               CompVision.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3, CompVision.LINE_AA) # Black outline
            CompVision.putText(highlight_frame, prompt_text, (10, 30), 
                               CompVision.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, CompVision.LINE_AA) # White text

            # Wait for user decision on this specific mask
            while True:
                CompVision.imshow("Segmented Output", highlight_frame)
                approval_key = CompVision.waitKey(0) & 0xFF # Wait indefinitely for a key press

                # DECISION: YES - Approve and save
                if approval_key == ord('y'):
                    print(f"  -> Approving mask {i+1}...")
                    bbox = mask_data['bbox']
                    x, y, w, h = bbox
                    cropped_object = frame[y:y+h, x:x+w]
                    cropped_mask = current_mask[y:y+h, x:x+w]
                    object_only = CompVision.bitwise_and(cropped_object, cropped_object, mask=cropped_mask.astype(np.uint8))

                    emb = get_dino_embedding(object_only)
                    if emb is not None:
                        normal_embeddings.append(emb)
                        current_total = len(normal_embeddings)
                        save_path = os.path.join(NORMAL_DATA_DIR, f"normal_{current_total}.jpg")
                        CompVision.imwrite(save_path, object_only)
                        print(f"     Saved. Total normal samples: {current_total}")
                    else:
                        print("     Failed to generate embedding.")
                    break # Move to the next mask

                # DECISION: NO - Reject
                elif approval_key == ord('n'):
                    print(f"  -> Rejecting mask {i+1}.")
                    break # Move to the next mask

                # DECISION: ESCAPE - Cancel entire approval process
                elif approval_key == 27: # Escape key
                    break # Break from inner while loop
                
            if approval_key == 27:
                print("--- CANCELLING APPROVAL MODE ---")
                break # Break from outer for loop
                
        print("--- APPROVAL MODE FINISHED, resuming live feed ---")

    elif key == ord('s'):
        if normal_embeddings:
            np.save(NORMAL_EMBEDDINGS_FILE, np.array(normal_embeddings, dtype=object))
            print(f"\n[ACTION] Saved {len(normal_embeddings)} total embeddings to {NORMAL_EMBEDDINGS_FILE}")
        else:
            print("\n[ACTION] No normal embeddings to save.")

    elif key == ord('q'):
        break

# --- Cleanup ---
print("Releasing resources...")
cap.release()
CompVision.destroyAllWindows()