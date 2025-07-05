cmake_minimum_required(VERSION 3.8)
project(lane_detection VERSION 0.1 LANGUAGES C CXX CUDA)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

#set(CMAKE_CUDA_FLAGS ${CMAKE_CUDA_FLAGS} "-g -G")

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

set(CUDA_MIN_VERSION "10.2")
set(CMAKE_CUDA_ARCHITECTURES 87)

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(pcl_conversions REQUIRED)
find_package(pcl_ros REQUIRED)
find_package(CUDA REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(OpenCV REQUIRED)
find_package(nav2_msgs REQUIRED)
find_package(nav2_util REQUIRED)
find_package(vizhi_interfaces REQUIRED)
#find_package(Eigen REQUIRED)
#find_package(Armadillo REQUIRED)
#find_package(OpenMP REQUIRED)
#find_package(mlpack REQUIRED)
find_package(Boost REQUIRED)


include_directories(
  include
  ${OpenCV_INCLUDE_DIRS}
  ${Boost_INCLUDE_DIRS}
  ${CUDA_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
  ${CMAKE_CURRENT_SOURCE_DIR}/src
  #${EIGEN3_INCLUDE_DIR}
  #${mlpack_INCLUDE_DIRS}
  #${armadillo_INCLUDE_DIRS}
)
# ament_python_install_package(${PROJECT_NAME})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

add_executable(cluster src/cluster.cu)
set_target_properties(cluster PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_compile_definitions(cluster
    PRIVATE
        # If the debug configuration pass the DEBUG define to the compiler 
        $<$<CONFIG:Debug>:DEBUG>
)
target_link_libraries(cluster
  ${OpenCV_LIBRARIES}
  ${Boost_LIBRARIES}
  ${CUDA_LIBRARIES}
  #${mlpack_LIBRARIES}
  #${armadillo_LIBRARIES}
)
target_compile_options(cluster PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
                       --use_fast_math
                       >)
ament_target_dependencies(cluster 
        rclcpp 
        sensor_msgs 
        cv_bridge 
        OpenCV 
        Boost 
        std_msgs 
        sensor_msgs 
        pcl_ros 
        pcl_conversions 
        vizhi_interfaces)


# add_executable(cluster_pointcloud src/cluster_pointcloud.cpp)
# ament_target_dependencies(cluster_pointcloud rclcpp sensor_msgs std_msgs pcl_ros pcl_conversions)
#add_executable(lane_image_publisher src/lane_image_publisher.cpp)
#target_link_libraries(lane_image_publisher
  #${OpenCV_LIBRARIES}
  ##${mlpack_LIBRARIES}
  #${armadillo_LIBRARIES}
  #${Boost_LIBRARIES}
#)
#ament_target_dependencies(lane_image_publisher 
	#rclcpp 
	#sensor_msgs 
	#cv_bridge 
	#OpenCV 
	#Boost 
	#std_msgs 
	#sensor_msgs 
	#pcl_ros 
	#pcl_conversions)
#install(
   #PROGRAMS 
   #DESTINATION lib/${PROJECT_NAME}
#)

add_executable(filter src/filter.cpp)
target_link_libraries(filter 
	${PCL_LIBRARIES} 
	${tf2_LIBRARIES})
ament_target_dependencies(filter 
	rclcpp 
	rclcpp_components 
	std_msgs 
	sensor_msgs 
	pcl_ros 
	pcl_conversions 
	tf2_ros 
	tf2_geometry_msgs)

install(TARGETS
  cluster
  filter
  # lane_image_publisher
  # cluster_pointcloud
  DESTINATION lib/${PROJECT_NAME})

install(DIRECTORY
  launch
  DESTINATION share/${PROJECT_NAME}/
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

include_directories(
  ${PCL_INCLUDE_DIRS}
)

ament_package()
