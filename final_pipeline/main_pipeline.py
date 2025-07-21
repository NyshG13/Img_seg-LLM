# main_pipeline.py

import os
import cv2
import numpy as np
from PIL import Image
import torch
import argparse
import time

# Import your project-specific modules
from grounded_sam2_tracking_camera_with_continuous_id import IncrementalObjectTracker
from qwen_analyzer import QwenVLAnalyzer
from report_generator import ReportGenerator

def main(args):
    # --- 1. Setup Environment ---
    output_dir = args.output_dir
    detected_objects_dir = os.path.join(output_dir, "detected_objects")
    os.makedirs(detected_objects_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # --- 2. Initialize Models ---
    print("Initializing models... This may take a moment.")
    tracker = IncrementalObjectTracker(
        grounding_model_id="IDEA-Research/grounding-dino-base",
        sam2_model_cfg="configs/sam2.1/sam2.1_hiera_l.yaml",
        sam2_ckpt_path="./checkpoints/sam2.1_hiera_large.pt",
        device=device,
        prompt_text=args.prompt,
        detection_interval=args.detection_interval,
    )
    
    analyzer = QwenVLAnalyzer(device=device)
    
    report_path = os.path.join(output_dir, "marsyard_anomaly_report.pdf")
    reporter = ReportGenerator(output_path=report_path)
    
    # --- 3. Setup Video Input ---
    video_source = args.video_path if args.video_path else 0
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        print(f"[Error] Cannot open video source: {video_source}")
        return

    print(f"[Info] Video source '{video_source}' opened. Press 'q' in the display window to quit.")
    
    # --- 4. Main Processing Loop ---
    frame_idx = 0
    analyzed_object_ids = set()

    # --- FPS Throttling Setup ---
    # Calculate the desired interval between frames in seconds. If fps is not set, interval is 0.
    frame_interval = 1.0 / args.fps if args.fps else 0
    last_processed_time = 0
    latest_annotated_frame = None # To hold the last processed frame for smooth display

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[Info] End of video stream or failed to capture frame.")
                break

            current_time = time.time()
            
            # --- Throttling Check ---
            # Check if enough time has passed to process a new frame.
            if (current_time - last_processed_time) < frame_interval:
                # If not enough time has passed, display the last processed frame and skip processing
                display_frame = latest_annotated_frame if latest_annotated_frame is not None else frame
                cv2.imshow("Live Pipeline", cv2.cvtColor(display_frame, cv2.COLOR_RGB2BGR) if latest_annotated_frame is not None else frame)
                if cv2.waitKey(1) & 0xFF == ord('q'): break
                continue
            
            # It's time to process a new frame, update the timestamp
            last_processed_time = current_time
            
            # --- Start of Frame Processing Block ---
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            print(f"\n--- Processing Frame {frame_idx} (Timestamp: {current_time:.2f}) ---")

            # 4.1. Run Grounded-SAM Tracking
            annotated_frame, mask_dict = tracker.add_image(frame_rgb)
            
            # Store the latest valid annotated frame
            if annotated_frame is not None:
                latest_annotated_frame = annotated_frame
            
            if mask_dict is None or not mask_dict.labels:
                print(f"[Warning] No objects tracked in frame {frame_idx}. Skipping analysis.")
                # Continue displaying the last successful frame or current raw frame
                display_frame = latest_annotated_frame if latest_annotated_frame is not None else frame
                cv2.imshow("Live Pipeline", cv2.cvtColor(display_frame, cv2.COLOR_RGB2BGR) if latest_annotated_frame is not None else frame)
                if cv2.waitKey(1) & 0xFF == ord('q'): break
                frame_idx += 1
                continue

            # 4.2. Process Newly Detected Objects
            current_frame_objects = mask_dict.labels
            for obj_id, obj_info in current_frame_objects.items():
                if obj_id not in analyzed_object_ids:
                    print(f"[New Object] Discovered object with ID: {obj_id}, Class: {obj_info.class_name}")
                    
                    x1, y1, x2, y2 = [int(c) for c in obj_info.box]
                    cropped_np = frame_rgb[y1:y2, x1:x2]
                    
                    if cropped_np.size == 0:
                        print(f"[Warning] Bounding box for object {obj_id} is empty. Skipping analysis.")
                        continue
                        
                    cropped_pil = Image.fromarray(cropped_np)
                    img_path = os.path.join(detected_objects_dir, f"object_{obj_id}_{obj_info.class_name}.jpg")
                    cropped_pil.save(img_path)
                    
                    print(f"[Qwen-VL] Analyzing object {obj_id}...")
                    description = analyzer.analyze_object(cropped_pil)
                    print(f"[Qwen-VL] Analysis complete for object {obj_id}.")
                    
                    reporter.add_object_entry(
                        object_id=obj_id,
                        class_name=obj_info.class_name,
                        image_path=img_path,
                        description=description
                    )
                    analyzed_object_ids.add(obj_id)

            # 4.3. Display Real-time Results
            display_frame = latest_annotated_frame if latest_annotated_frame is not None else frame
            display_bgr = cv2.cvtColor(display_frame, cv2.COLOR_RGB2BGR) if latest_annotated_frame is not None else frame
            cv2.imshow("Live Pipeline", display_bgr)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("[Info] Quit signal received.")
                break
            
            frame_idx += 1
            if args.max_frames and frame_idx >= args.max_frames:
                print(f"[Info] Reached max_frames limit of {args.max_frames}.")
                break
                
    except KeyboardInterrupt:
        print("\n[Info] Interrupted by user (Ctrl+C).")
    finally:
        # --- 5. Finalize and Cleanup ---
        print("\n--- Finalizing Pipeline ---")
        reporter.save_report()
        cap.release()
        cv2.destroyAllWindows()
        print("[Done] Pipeline finished.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the full Grounded-SAM + Qwen-VL pipeline.")
    parser.add_argument("--video_path", type=str, default=None, help="Path to a .mp4 video file. If not provided, webcam will be used.")
    parser.add_argument("--prompt", type=str, default="object.", help="Text prompt for GroundingDINO.")
    parser.add_argument("--output_dir", type=str, default="./outputs", help="Directory to save the report and detected objects.")
    parser.add_argument("--detection_interval", type=int, default=10, help="Frame interval for running the object detector.")
    parser.add_argument("--max_frames", type=int, default=None, help="Maximum number of frames to process from the video.")
    # --- New Argument ---
    parser.add_argument("--fps", type=float, default=None, help="Target processing FPS to throttle the pipeline. If not set, processes frames as fast as possible.")
    
    args = parser.parse_args()
    main(args)