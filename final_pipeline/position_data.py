#!/usr/bin/env python3
import rospy
import numpy as np
from sensor_msgs.msg import NavSatFix, Imu, Image
from cv_bridge import CvBridge
import tf
import math

# === Global variables updated by callbacks ===
latest_gps = None
latest_yaw_deg = None
latest_depth_frame = None

bridge = CvBridge()

# === GPS Callback ===
def gps_callback(msg):
    global latest_gps
    latest_gps = (msg.latitude, msg.longitude)

# === IMU Callback (yaw in degrees) ===
def imu_callback(msg):
    global latest_yaw_deg
    orientation_q = msg.orientation
    quaternion = (orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w)
    euler = tf.transformations.euler_from_quaternion(quaternion)
    yaw_rad = euler[2]
    latest_yaw_deg = math.degrees(yaw_rad)

# === Depth Image Callback ===
def depth_callback(msg):
    global latest_depth_frame
    try:
        depth_image = bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        latest_depth_frame = np.array(depth_image, dtype=np.float32)
    except Exception as e:
        rospy.logerr(f"Depth conversion error: {e}")

def get_current_gps_position():
    if latest_gps:
        return latest_gps
    else:
        rospy.logwarn("GPS data not available yet.")
        return None

def get_current_yaw():
    if latest_yaw_deg is not None:
        return latest_yaw_deg
    else:
        rospy.logwarn("IMU yaw data not available yet.")
        return None

def get_depth_for_pixel(x, y):
    if latest_depth_frame is None:
        rospy.logwarn("Depth frame not available.")
        return None
    h, w = latest_depth_frame.shape
    if 0 <= x < w and 0 <= y < h:
        depth = latest_depth_frame[y, x]
        return float(depth) if not np.isnan(depth) else None
    else:
        rospy.logwarn("Pixel out of bounds.")
        return None

if __name__ == "__main__":
    rospy.init_node("sensor_fetcher")

    rospy.Subscriber("/gps/fix", NavSatFix, gps_callback)
    rospy.Subscriber("/imu/data", Imu, imu_callback)
    rospy.Subscriber("/camera/depth/image_raw", Image, depth_callback)

    rospy.loginfo("Sensor node running...")
    rospy.spin()  # Keeps the node alive
