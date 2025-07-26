cmake_minimum_required(VERSION 3.8)
project(final_pipeline)

# Dependencies
find_package(ament_cmake REQUIRED)
find_package(rclpy REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)

# For service files
find_package(rosidl_default_generators REQUIRED)

# Service files
rosidl_generate_interfaces(${PROJECT_NAME}
  "srv/GetGPS.srv"
  "srv/GetYaw.srv"
  "srv/GetDepth.srv"
  DEPENDENCIES std_msgs
)

ament_export_dependencies(rosidl_default_runtime)

# Install executables (for ros2 run)
install(PROGRAMS
  final_pipeline/new_pipeline.py
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()

