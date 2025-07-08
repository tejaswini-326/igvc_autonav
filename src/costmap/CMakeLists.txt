cmake_minimum_required(VERSION 3.8)
project(costmap)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_sensor_msgs REQUIRED)
find_package(OpenCV REQUIRED)
# uncomment the following section in order to fill in
# further dependencies manually.
# find_package(<dependency> REQUIRED)
include_directories(
  include
  ${OpenCV_INCLUDE_DIRS}
)

add_executable(costmap_node src/costmap_node.cpp)

ament_target_dependencies(costmap_node
  rclcpp
  sensor_msgs
  nav_msgs
  tf2
  tf2_ros
  tf2_sensor_msgs
  OpenCV
)

target_link_libraries(costmap_node
  ${OpenCV_LIBS}
)

install(TARGETS costmap_node
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()