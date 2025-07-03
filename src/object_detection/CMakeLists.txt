cmake_minimum_required(VERSION 3.8)
project(object_detection)

# Enable stricter compiler warnings
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(builtin_interfaces REQUIRED)
find_package(rosidl_default_generators REQUIRED)

# Message generation
rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/ObjectData.msg"
  DEPENDENCIES geometry_msgs sensor_msgs std_msgs builtin_interfaces
)

install(PROGRAMS
  src/object_data_node.py
  DESTINATION lib/${PROJECT_NAME}
)

install(PROGRAMS
  src/back_object_data_node.py
  DESTINATION lib/${PROJECT_NAME}
)
# Testing setup
if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

# Needed for downstream packages and runtime
ament_export_dependencies(rosidl_default_runtime)
ament_package()
