#!/usr/bin/env python3
import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution, Command
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

# 0 = concrete.world
# 1 = concrete_overdose.world
LAUNCH_TYPE = 1

if LAUNCH_TYPE == 0:
    world_file = "concrete.world"
    x_init, y_init, yaw_init = '-24.580000', '26.260000', '-1.8'
elif LAUNCH_TYPE == 1:
    world_file = 'concrete_overdose.world'
    #x_init, y_init, yaw_init = "22.210000", "19.332100", "1.573740"
    x_init, y_init, yaw_init = '-3.309640', '28.355800', '2.564250'  
    #x_init, y_init, yaw_init = '-26.193200', '-25.706900', '-0.630470'


def generate_launch_description():
    # ------------------------------------------------------------------------
    # Locate your package and set up Gazebo resource paths
    # ------------------------------------------------------------------------
    pkg_igvc = get_package_share_directory('igvc')
    # We want Gazebo to find both the worlds/ and models/ directories
    gazebo_models_path, _ = os.path.split(pkg_igvc)
    os.environ["GZ_SIM_RESOURCE_PATH"] += os.pathsep + gazebo_models_path

    # ------------------------------------------------------------------------
    # Declare launch arguments (with defaults and descriptions)
    # ------------------------------------------------------------------------
    rviz_launch_arg = DeclareLaunchArgument(
        'rviz',
        default_value='true',
        description='Whether to start RViz'
    )
    rviz_config_arg = DeclareLaunchArgument(
        'rviz_config',
        default_value='rviz.rviz',
        description='RViz configuration file name (in rviz/ folder)'
    )
    world_arg = DeclareLaunchArgument(
        'world',
        default_value=world_file,
        description='Name of the Ignition world file to load'
    )
    model_arg = DeclareLaunchArgument(
        'model',
        default_value='mogi_bot.urdf',
        description='Name of the URDF (or Xacro) file to spawn'
    )




    x_arg = DeclareLaunchArgument(
        'x',
        default_value=x_init, 
        description='Initial X coordinate for robot spawn'
    )
    y_arg = DeclareLaunchArgument(
        'y',
        default_value=y_init,
        description='Initial Y coordinate for robot spawn'
    )
    yaw_arg = DeclareLaunchArgument(
        'yaw',
        default_value=yaw_init,
        description='Initial yaw (rotation around Z) for robot spawn'
    )

    # x_arg = DeclareLaunchArgument(
    #     'x',
    #     default_value='-26.217200',
    #     description='Initial X coordinate for robot spawn'
    # )
    # y_arg = DeclareLaunchArgument(
    #     'y',
    #     default_value='-8.613460',
    #     description='Initial Y coordinate for robot spawn'
    # )
    # yaw_arg = DeclareLaunchArgument(
    #     'yaw',
    #     default_value='-1.57',
    #     description='Initial yaw (rotation around Z) for robot spawn'
    # )



    # x_arg = DeclareLaunchArgument(
    #     'x',
    #     default_value='-23.061300',
    #     description='Initial X coordinate for robot spawn'
    # )
    # y_arg = DeclareLaunchArgument(
    #     'y',
    #     default_value='-27.155000',
    #     description='Initial Y coordinate for robot spawn'
    # )
    # yaw_arg = DeclareLaunchArgument(
    #     'yaw',
    #     default_value='-0.400308',
    #     description='Initial yaw (rotation around Z) for robot spawn'
    # )


    sim_time_arg = DeclareLaunchArgument(
        'use_sim_time',
        default_value='True',
        description='Enable /clock simulation time'
    )

    # ------------------------------------------------------------------------
    # Compose the path to the URDF (or xacro) file
    # ------------------------------------------------------------------------
    urdf_file_path = PathJoinSubstitution([
        pkg_igvc,
        "urdf",
        LaunchConfiguration('model')
    ])

    # ------------------------------------------------------------------------
    # Include the world-launch file (starts Gazebo with the given world)
    # ------------------------------------------------------------------------
    world_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_igvc, 'launch', 'igvc_world.launch.py'),
        ),
        launch_arguments={
            'world': LaunchConfiguration('world'),
        }.items()
    )

    # ------------------------------------------------------------------------
    # RViz node (runs only if --ros-args -p rviz:=true)
    # ------------------------------------------------------------------------
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        arguments=[
            '-d',
            PathJoinSubstitution([
                pkg_igvc,
                'rviz',
                LaunchConfiguration('rviz_config')
            ])
        ],
        condition=IfCondition(LaunchConfiguration('rviz')),
        parameters=[{'use_sim_time': LaunchConfiguration('use_sim_time')}],
    )

    pointcloud_downscaler_node = Node(
        package='movement',
        executable='pointcloud_downscaler',
        name='PointCloudDownscaler',
        output='screen',
        parameters=[{'use_sim_time': LaunchConfiguration('use_sim_time')}],
    )

    back_pointcloud_downscaler_node = Node(
        package='movement',
        executable='back_pointcloud_downscaler',
        name='BackPointCloudDownscaler',
        output='screen',
        parameters=[{'use_sim_time': LaunchConfiguration('use_sim_time')}],
    )

    gps_waypoint_publisher_node = Node(
        package='movement',
        executable='gps_waypoint_publisher',
        name='GPSNextWaypointPublisherNode',
        output='screen',
        parameters=[{'use_sim_time': LaunchConfiguration('use_sim_time')}],
    )



    goal_publisher_node = Node(
        package='movement',
        executable='goal_publisher',
        name='goal_publisher',
        parameters=[{'use_sim_time': LaunchConfiguration('use_sim_time')}],
        output='screen'
    )

    costmap_publisher_node = Node(
        package='navigation',
        executable='costmap',
        name='costmap_publisher',
        parameters=[{'use_sim_time': LaunchConfiguration('use_sim_time')}],
        output='screen'
    )

    path_publisher_node = Node(
        package='path_planning',
        executable='path_planner',
        name='path_planner',
        parameters=[{'use_sim_time': LaunchConfiguration('use_sim_time')}],
        output='screen'
    )

    curve_fit_node = Node(
        package='navigation',
        executable='curve_fit',
        name='lane_fit',
        parameters=[{'use_sim_time': LaunchConfiguration('use_sim_time')}],
        output='screen'
    )
    controller_node = Node(
        package='navigation',
        executable='controller',
        name='controller',
        parameters=[{'use_sim_time': LaunchConfiguration('use_sim_time')}],
        output='screen'
    )
    m_controller_node = Node(
        package='navigation',
        executable='m_controller',
        name='m_controller',
        parameters=[{'use_sim_time': LaunchConfiguration('use_sim_time')}],
        output='screen'
    )

    lane_direction_finder_node = Node(
        package='navigation',
        executable='lane_direction_finder',
        name='lane_direction_finder',
        parameters=[{'use_sim_time': LaunchConfiguration('use_sim_time')}],
        output='screen'
    )

    lidar_node = Node(
        package='navigation',  
        executable='lidar',  
        name='scan_to_pointcloud',
        output='screen',
        parameters=[{'use_sim_time': LaunchConfiguration('use_sim_time')}],
    )
   

    # ------------------------------------------------------------------------
    # Spawn the robot into Gazebo via the /world/.../create service
    # ------------------------------------------------------------------------
    spawn_urdf_node = Node(
        package="ros_gz_sim",
        executable="create",
        arguments=[
            "-name", "mogi_bot",                     # the model name
            "-topic", "robot_description",           # reads from this ROS topic
            "-x", LaunchConfiguration('x'),
            "-y", LaunchConfiguration('y'),
            "-z", "0.5",                             # fixed Z height
            "-Y", LaunchConfiguration('yaw'),
        ],
        output="screen",
        parameters=[{'use_sim_time': LaunchConfiguration('use_sim_time')}],
    )

    # ------------------------------------------------------------------------
    # Bridge common topics between ROS 2 and Gazebo
    # ------------------------------------------------------------------------
    gz_bridge_node = Node(
        package="ros_gz_bridge",
        executable="parameter_bridge",
        arguments=[
            "/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock",
            "/cmd_vel@geometry_msgs/msg/Twist@gz.msgs.Twist",
            "/odom@nav_msgs/msg/Odometry@gz.msgs.Odometry",
            "/joint_states@sensor_msgs/msg/JointState@gz.msgs.Model",
            "/camera/camera_info@sensor_msgs/msg/CameraInfo@gz.msgs.CameraInfo",
            "/imu@sensor_msgs/msg/Imu@gz.msgs.IMU",
            "/navsat@sensor_msgs/msg/NavSatFix@gz.msgs.NavSat",
            "/scan@sensor_msgs/msg/LaserScan@gz.msgs.LaserScan",
            "/scan/points@sensor_msgs/msg/PointCloud2@gz.msgs.PointCloudPacked",
            "/camera/depth_image@sensor_msgs/msg/Image@gz.msgs.Image",
            "/camera/points@sensor_msgs/msg/PointCloud2@gz.msgs.PointCloudPacked",
            "/bcamera/camera_info@sensor_msgs/msg/CameraInfo@gz.msgs.CameraInfo",
            "/bcamera/depth_image@sensor_msgs/msg/Image@gz.msgs.Image",
            "/bcamera/points@sensor_msgs/msg/PointCloud2@gz.msgs.PointCloudPacked",
            "/tf@tf2_msgs/msg/TFMessage[gz.msgs.Pose_V",
            #"/tf_static@tf2_msgs/msg/TFMessage[gz.msgs.Pose_V",
        ],
        output="screen",
        parameters=[{'use_sim_time': LaunchConfiguration('use_sim_time')}],
    )

    # ------------------------------------------------------------------------
    # Image bridge for camera with compressed transport
    # ------------------------------------------------------------------------
    gz_image_bridge_node = Node(
        package="ros_gz_image",
        executable="image_bridge",
        arguments=["/camera/image"],
        output="screen",
        parameters=[{
            'use_sim_time': LaunchConfiguration('use_sim_time'),
            #'camera.image.compressed.jpeg_quality': 75
        }],
    )   

    # ------------------------------------------------------------------------
    # Relay camera_info under the proper topic namespace
    # ------------------------------------------------------------------------
    relay_camera_info_node = Node(
        package='topic_tools',
        executable='relay',
        name='relay_camera_info',
        output='screen',
        arguments=['camera/camera_info'],
        parameters=[{'use_sim_time': LaunchConfiguration('use_sim_time')}],
    )

    gz_bimage_bridge_node = Node(
        package="ros_gz_image",
        executable="image_bridge",
        arguments=["/bcamera/image"],
        output="screen",
        parameters=[{
            'use_sim_time': LaunchConfiguration('use_sim_time'),
            #'camera.image.compressed.jpeg_quality': 75
        }],
    )   

    relay_bcamera_info_node = Node(
        package='topic_tools',
        executable='relay',
        name='relay_bcamera_info',
        output='screen',
        arguments=['bcamera/camera_info', 'bcamera/image/camera_info'],
        parameters=[{'use_sim_time': LaunchConfiguration('use_sim_time')}],
    )



    # ------------------------------------------------------------------------
    # EKF node for sensor fusion (robot_localization)
    # ------------------------------------------------------------------------
    ekf_node = Node(
        package='robot_localization',
        executable='ekf_node',
        name='ekf_filter_node',
        output='screen',
        parameters=[
            os.path.join(pkg_igvc, 'config', 'ekf.yaml'),
            {'use_sim_time': LaunchConfiguration('use_sim_time')}
        ],
    )

    # ------------------------------------------------------------------------
    # Robot State Publisher (publishes TF from the robot_description)
    # ------------------------------------------------------------------------
    robot_state_publisher_node = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        output='screen',
        parameters=[{
            'robot_description': Command(['xacro ', urdf_file_path]),
            'use_sim_time': LaunchConfiguration('use_sim_time')
        }],
    )


    # ------------------------------------------------------------------------
    # Assemble and return the LaunchDescription
    # ------------------------------------------------------------------------
    ld = LaunchDescription()

    # Add all declared arguments
    ld.add_action(rviz_launch_arg)
    ld.add_action(rviz_config_arg)
    ld.add_action(world_arg)
    ld.add_action(model_arg)
    ld.add_action(x_arg)
    ld.add_action(y_arg)
    ld.add_action(yaw_arg)
    ld.add_action(sim_time_arg)

    # Add all nodes and included launches
    ld.add_action(world_launch)
    ld.add_action(rviz_node)
    ld.add_action(spawn_urdf_node)

    ld.add_action(gz_bridge_node)
    ld.add_action(gz_image_bridge_node)
    ld.add_action(gz_bimage_bridge_node)
    ld.add_action(relay_camera_info_node)
    ld.add_action(relay_bcamera_info_node)

    ld.add_action(ekf_node)
    ld.add_action(robot_state_publisher_node)
    
    ld.add_action(gps_waypoint_publisher_node)
    ld.add_action(pointcloud_downscaler_node)
    ld.add_action(back_pointcloud_downscaler_node)
    ld.add_action(m_controller_node)
    ld.add_action(lane_direction_finder_node)


    ld.add_action(costmap_publisher_node)
    ld.add_action(curve_fit_node)
    ld.add_action(lidar_node)

    return ld

