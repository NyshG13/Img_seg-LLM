cmake_minimum_required(VERSION 3.8)
project(final_pipeline_interfaces)

# Find necessary packages
find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)

# This is the key command to generate code from your .srv files.
# List all your .srv files here.
rosidl_generate_interfaces(${PROJECT_NAME}
  "srv/GetDepth.srv"
  "srv/GetGPS.srv"
  "srv/GetYaw.srv"
  # "msg/YourMessage.msg"  <- Add msg files here if you have them
)

ament_package()