# main_pipeline.py

import os
import cv2
import numpy as np
from PIL import Image
import torch
import argparse
import time
import sys
from ament_index_python.packages import get_package_share_directory

# Import your project-specific modules
from extras.grounded_sam2_tracking_camera_with_continuous_id import IncrementalObjectTracker
from extras.qwen_analyzer import QwenVLAnalyzer
from extras.report_generator import ReportGenerator
from gps_tracking import pixel_to_gps
from gps_tracking import handle_detection

# ROS2 Imports
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image as RosImage
from cv_bridge import CvBridge
from final_pipeline.srv import GetGPS, GetYaw, GetDepth  #this works only after colcon building the package

class PerceptionPipelineNode(Node):
    def __init__(self, args):
        """
        Initializes the ROS2 node, models, and all necessary components for the perception pipeline.
        """
        super().__init__('perception_pipeline_node')
        self.args = args

        # --- 1. Setup Environment ---
        self.output_dir = self.args.output_dir
        self.detected_objects_dir = os.path.join(self.output_dir, "detected_objects")
        os.makedirs(self.detected_objects_dir, exist_ok=True)
        self.get_logger().info(f"Output directory set to: {self.output_dir}")

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.get_logger().info(f"Using device: {self.device}")

        BASE_PATH = os.path.dirname(__file__)  # Points to final_pipeline/final_pipeline/
        sam2_ckpt_path = os.path.join(BASE_PATH, 'sam2', 'checkpoints', 'sam2.1_hiera_large.pt')
        sam2_model_cfg = os.path.join(BASE_PATH, 'sam2', 'configs', 'sam2.1', 'sam2.1_hiera_l.yaml')


        # --- 2. Initialize Models ---
        self.get_logger().info("Initializing models... This may take a moment.")
        self.tracker = IncrementalObjectTracker(
            grounding_model_id="IDEA-Research/grounding-dino-base",
            sam2_model_cfg=sam2_model_cfg,
            sam2_ckpt_path=sam2_ckpt_path,
            device=self.device,
            prompt_text=self.args.prompt,
            detection_interval=self.args.detection_interval,
        )
        self.analyzer = QwenVLAnalyzer(device=self.device)
        report_path = os.path.join(self.output_dir, "marsyard_anomaly_report.pdf")
        self.reporter = ReportGenerator(output_path=report_path)
        self.get_logger().info("Models initialized successfully.")

        # --- 3. Setup ROS2 Connections ---
        # CV Bridge for converting ROS Image messages to OpenCV format
        self.bridge = CvBridge()

        # Service Clients
        self.get_logger().info("Waiting for ROS services...")
        self.gps_client = self.create_client(GetGPS, 'get_gps')
        self.yaw_client = self.create_client(GetYaw, 'get_yaw')
        self.depth_client = self.create_client(GetDepth, 'get_depth')

        for client in [self.gps_client, self.yaw_client, self.depth_client]:
            while not client.wait_for_service(timeout_sec=2.0):
                self.get_logger().info(f'Service "{client.srv_name}" not available, waiting...')
        self.get_logger().info("All services are available.")

        # Camera Subscriber
        self.subscription = self.create_subscription(
            RosImage,
            self.args.camera_topic,
            self.image_callback,
            10  # QoS profile depth
        )
        self.get_logger().info(f"Subscribed to camera topic: '{self.args.camera_topic}'")

        # --- 4. Initialize State Variables ---
        self.frame_idx = 0
        self.analyzed_object_ids = set()
        self.frame_interval = 1.0 / self.args.fps if self.args.fps else 0
        self.last_processed_time = 0
        self.latest_annotated_frame = None

    def image_callback(self, msg):
        """
        This function is called for every frame received from the camera topic.
        It contains the main processing logic.
        """
        current_time = time.time()
        
        # --- Throttling Check ---
        if (current_time - self.last_processed_time) < self.frame_interval:
            return  # Skip processing this frame to maintain target FPS

        self.last_processed_time = current_time

        # Check for max_frames limit
        if self.args.max_frames and self.frame_idx >= self.args.max_frames:
            if self.frame_idx == self.args.max_frames: # Log only once
                 self.get_logger().info(f"Reached max_frames limit of {self.args.max_frames}. No longer processing frames.")
                 self.frame_idx += 1 # Increment to prevent re-logging
            return

        try:
            # Convert ROS Image message to OpenCV image (BGR format)
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f"Failed to convert ROS Image to OpenCV format: {e}")
            return

        # --- Start of Frame Processing Block ---
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.get_logger().info(f"\n--- Processing Frame {self.frame_idx} (Timestamp: {current_time:.2f}) ---")

        # 4.1. Run Grounded-SAM Tracking
        annotated_frame, mask_dict = self.tracker.add_image(frame_rgb)
        
        if annotated_frame is not None:
            self.latest_annotated_frame = annotated_frame
        
        if mask_dict is None or not mask_dict.labels:
            self.get_logger().warn(f"No objects tracked in frame {self.frame_idx}. Skipping analysis.")
        else:
            # 4.2. Process Newly Detected Objects
            current_frame_objects = mask_dict.labels
            for obj_id, obj_info in current_frame_objects.items():
                if obj_id not in self.analyzed_object_ids:
                    self.get_logger().info(f"[New Object] Discovered object with ID: {obj_id}, Class: {obj_info.class_name}")
                    
                    x1, y1, x2, y2 = [int(c) for c in obj_info.box]
                    pixel_x = int((x1 + x2) / 2)
                    pixel_y = int((y1 + y2) / 2)

                    cropped_np = frame_rgb[y1:y2, x1:x2]
                    
                    if cropped_np.size == 0:
                        self.get_logger().warn(f"Bounding box for object {obj_id} is empty. Skipping.")
                        continue
                        
                    cropped_pil = Image.fromarray(cropped_np)

                    # === Get current data from ROS services ===
                    depth_value = self.get_depth_at_pixel(pixel_x, pixel_y)
                    if depth_value is None or np.isnan(depth_value) or depth_value <= 0:
                        self.get_logger().warn(f"Invalid depth at pixel ({pixel_x},{pixel_y}). Skipping object.")
                        continue

                    gps_pos = self.get_rover_gps()
                    rover_yaw = self.get_rover_yaw()
                    if gps_pos is None or rover_yaw is None:
                        self.get_logger().error("Failed to get rover pose (GPS/Yaw). Skipping object.")
                        continue

                    # === Convert to GPS ===
                    lat, lon = pixel_to_gps(pixel_x, pixel_y, depth_value, gps_pos.latitude, gps_pos.longitude, rover_yaw)
                    send_to_qwen = handle_detection(lat, lon, "weird_objects")

                    img_path = os.path.join(self.detected_objects_dir, f"object_{obj_id}_{obj_info.class_name}.jpg")
                    cropped_pil.save(img_path)
                    
                    if not send_to_qwen:
                        self.get_logger().warn(f"Qwen-VL analysis skipped for object {obj_id} by handle_detection logic.")
                        continue

                    self.get_logger().info(f"[Qwen-VL] Analyzing object {obj_id}...")
                    description = self.analyzer.analyze_object(cropped_pil)
                    self.get_logger().info(f"[Qwen-VL] Analysis complete for object {obj_id}.")
                    
                    self.reporter.add_object_entry(
                        object_id=obj_id,
                        class_name=obj_info.class_name,
                        image_path=img_path,
                        description=description
                    )
                    self.analyzed_object_ids.add(obj_id)

        # 4.3. Display Real-time Results
        display_frame = self.latest_annotated_frame if self.latest_annotated_frame is not None else frame
        display_bgr = cv2.cvtColor(display_frame, cv2.COLOR_RGB2BGR) if self.latest_annotated_frame is not None and display_frame.shape[2] == 3 else frame
        cv2.imshow("Live Pipeline", display_bgr)
        cv2.waitKey(1) # Necessary for the window to update
        
        self.frame_idx += 1

    # --- Helper methods for calling ROS Services ---
    def get_depth_at_pixel(self, x, y):
        req = GetDepth.Request()
        req.x = x
        req.y = y
        future = self.depth_client.call_async(req)
        rclpy.spin_until_future_complete(self, future, timeout_sec=1.0)
        if future.result() is not None:
            return future.result().depth
        self.get_logger().error(f"Service call to get_depth failed: {future.exception()}")
        return None

    def get_rover_gps(self):
        req = GetGPS.Request()
        future = self.gps_client.call_async(req)
        rclpy.spin_until_future_complete(self, future, timeout_sec=1.0)
        if future.result() is not None:
            return future.result()
        self.get_logger().error(f"Service call to get_gps failed: {future.exception()}")
        return None

    def get_rover_yaw(self):
        req = GetYaw.Request()
        future = self.yaw_client.call_async(req)
        rclpy.spin_until_future_complete(self, future, timeout_sec=1.0)
        if future.result() is not None:
            return future.result().yaw_deg
        self.get_logger().error(f"Service call to get_yaw failed: {future.exception()}")
        return None

    def on_shutdown(self):
        """
        Cleanup operations when the node is shut down.
        """
        self.get_logger().info("\n--- Finalizing Pipeline ---")
        self.reporter.save_report()
        cv2.destroyAllWindows()
        self.get_logger().info("[Done] Pipeline finished.")


def main(cli_args=None):
    if cli_args is None:
        cli_args = sys.argv

    parser = argparse.ArgumentParser(description="Run the full Grounded-SAM + Qwen-VL pipeline on a ROS2 camera topic.")
    parser.add_argument("--camera_topic", type=str, default="/image_raw", help="ROS2 topic for the input camera stream.")
    parser.add_argument("--prompt", type=str, default="object.", help="Text prompt for GroundingDINO.")
    parser.add_argument("--output_dir", type=str, default="./outputs", help="Directory to save the report and detected objects.")
    parser.add_argument("--detection_interval", type=int, default=10, help="Frame interval for running the object detector.")
    parser.add_argument("--max_frames", type=int, default=None, help="Maximum number of frames to process.")
    parser.add_argument("--fps", type=float, default=5.0, help="Target processing FPS to throttle the pipeline. If not set, processes frames as fast as possible.")
    
    args = parser.parse_args(cli_args[1:]) # Skip the program name

    rclpy.init(args=cli_args)
    pipeline_node = None
    try:
        pipeline_node = PerceptionPipelineNode(args)
        rclpy.spin(pipeline_node)
    except KeyboardInterrupt:
        print("\n[Info] Interrupted by user (Ctrl+C). Shutting down.")
    except Exception as e:
        if pipeline_node:
            pipeline_node.get_logger().fatal(f"An unhandled exception occurred: {e}", exc_info=True)
        else:
            print(f"An unhandled exception occurred during node initialization: {e}")
    finally:
        if pipeline_node:
            pipeline_node.on_shutdown()
            pipeline_node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == "__main__":
    main()