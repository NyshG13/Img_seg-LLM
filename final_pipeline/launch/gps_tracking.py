import uuid 
from datetime import datetime
from geopy.distance import geodesic
import numpy as np
from math import cos, sin, radians
from datetime import timezone

def pixel_to_gps(pixel_x, pixel_y, depth_meters, 
                 rover_lat, rover_lon, rover_yaw_deg,
                 fx=600, fy=600, cx=320, cy=240):
    """
    Converts camera pixel coordinates and depth to GPS coordinates.
    
    Args:
        pixel_x, pixel_y: 2D pixel location of object in image
        depth_meters: distance to object (from LiDAR)
        rover_lat, rover_lon: current GPS position of rover
        rover_yaw_deg: orientation of rover in degrees (0 = North, 90 = East)
        fx, fy: focal lengths of camera
        cx, cy: principal point (center of image)
    
    Returns:
        (object_lat, object_lon): Estimated GPS position of the object
    """

        # === Step 1: Pixel to 3D point in camera frame ===
    x_cam = (pixel_x - cx) * depth_meters / fx
    y_cam = (pixel_y - cy) * depth_meters / fy
    z_cam = depth_meters
    point_cam = np.array([x_cam, y_cam, z_cam])

    yaw_rad = radians(rover_yaw_deg)
    R = np.array([
        [cos(yaw_rad), -sin(yaw_rad), 0],
        [sin(yaw_rad),  cos(yaw_rad), 0],
        [0,             0,            1]
    ])
    point_world = R @ point_cam

    # === Step 3: Convert displacement to lat/lon offset ===
    delta_east_m = point_world[0]
    delta_north_m = point_world[2]

    lat_offset = delta_north_m / 111000  # meters to degrees
    lon_offset = delta_east_m / (111000 * cos(radians(rover_lat)))

    lat = rover_lat + lat_offset
    lon = rover_lon + lon_offset

    return lat, lon

# === Part 1: Initialize Geofence Database ===
geofence_db = []

# === Part 2: Helper Function to Check Radius ===
def is_duplicate(new_coord, threshold=2.0):  # meters
    for entry in geofence_db:
        dist = geodesic(entry["global_coordinates"], new_coord).meters
        if dist <= threshold:
            return True
    return False

# === Part 3: Add New Object ===
def add_object_to_db(lat, lon, obj_class):
    obj = {
        "object_id": str(uuid.uuid4()),
        "global_coordinates": (lat, lon),
        "last_reported_timestamp": datetime.now(datetime.timezone.utc).isoformat(),
        "object_class": obj_class
    }
    geofence_db.append(obj)
    print(f"Added new object to DB: {obj}")


# === Part 4: Use Case ===
# This function is called when a new object is detected
def handle_detection(lat, lon, obj_class):
    if is_duplicate((lat, lon)):
        print("Duplicate detected. Skipping...")
        return False
    else:
        # send cropped image to Qwen or save JSON etc.
        add_object_to_db(lat, lon, obj_class)
        return True 


#this stores that object as weird object after making sure it hasnt been seen before

# lat, lon = pixel_to_gps(
#     pixel_x=100, pixel_y=100,          # Object's location in image (in pixels)
#     depth_meters=100,                  # Distance from rover to object (LiDAR)
#     rover_lat=28.6139,                 # Current GPS latitude of rover
#     rover_lon=77.2090,                 # Current GPS longitude of rover
#     rover_yaw_deg=0                    # Direction the rover is facing (0° = North)
# )

# pixel_x and pixel_y --> from gdino file 
# rover lat, lon, yaw--> from gps file (.msg)
# depth_meters --> from lidar file
