#!/usr/bin/env python3
import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution, Command
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    # ------------------------------------------------------------------------
    # Locate your package and set up Gazebo resource paths
    # ------------------------------------------------------------------------
    pkg_bme_gazebo_sensors = get_package_share_directory('bme_gazebo_sensors')
    # We want Gazebo to find both the worlds/ and models/ directories
    gazebo_models_path, _ = os.path.split(pkg_bme_gazebo_sensors)
    os.environ["GZ_SIM_RESOURCE_PATH"] += os.pathsep + gazebo_models_path

    # ------------------------------------------------------------------------
    # Declare launch arguments (with defaults and descriptions)
    # ------------------------------------------------------------------------
    rviz_launch_arg = DeclareLaunchArgument(
        'rviz',
        default_value='false',
        description='Whether to start RViz'
    )
    rviz_config_arg = DeclareLaunchArgument(
        'rviz_config',
        default_value='rviz.rviz',
        description='RViz configuration file name (in rviz/ folder)'
    )
    world_arg = DeclareLaunchArgument(
        'world',
        default_value='asphault.world',
        description='Name of the Ignition world file to load'
    )
    model_arg = DeclareLaunchArgument(
        'model',
        default_value='mogi_bot.urdf',
        description='Name of the URDF (or Xacro) file to spawn'
    )
    x_arg = DeclareLaunchArgument(
        'x',
        default_value='-30.10',
        description='Initial X coordinate for robot spawn'
    )
    y_arg = DeclareLaunchArgument(
        'y',
        default_value='-2.0',
        description='Initial Y coordinate for robot spawn'
    )
    yaw_arg = DeclareLaunchArgument(
        'yaw',
        default_value='-1.5707',
        description='Initial yaw (rotation around Z) for robot spawn'
    )
    sim_time_arg = DeclareLaunchArgument(
        'use_sim_time',
        default_value='True',
        description='Enable /clock simulation time'
    )

    # ------------------------------------------------------------------------
    # Compose the path to the URDF (or xacro) file
    # ------------------------------------------------------------------------
    urdf_file_path = PathJoinSubstitution([
        pkg_bme_gazebo_sensors,
        "urdf",
        LaunchConfiguration('model')
    ])

    # ------------------------------------------------------------------------
    # Include the world-launch file (starts Gazebo with the given world)
    # ------------------------------------------------------------------------
    world_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_bme_gazebo_sensors, 'launch', 'igvc_world.launch.py'),
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
                pkg_bme_gazebo_sensors,
                'rviz',
                LaunchConfiguration('rviz_config')
            ])
        ],
        condition=IfCondition(LaunchConfiguration('rviz')),
        parameters=[{'use_sim_time': LaunchConfiguration('use_sim_time')}],
    )

    intersection_straight_node = Node(
        package='bme_gazebo_sensors_py',
        executable='intersection_straight',
        name='IntersectionStraightDriver',
        output='screen',
        parameters=[{'use_sim_time': LaunchConfiguration('use_sim_time')}],
    )
    
    intersection_left_node = Node(
        package='bme_gazebo_sensors_py',
        executable='intersection_left',
        name='IntersectionLeftTurnDriver',
        output='screen',
        parameters=[{'use_sim_time': LaunchConfiguration('use_sim_time')}],
    )

    gps_waypoint_publisher_node = Node(
        package='bme_gazebo_sensors_py',
        executable='gps_waypoint_publisher',
        name='GPSNextWaypointPublisherNode',
        output='screen',
        parameters=[{'use_sim_time': LaunchConfiguration('use_sim_time')}],
    )

    move_forward_node = Node(
        package='bme_gazebo_sensors_py',
        executable='move_forward',
        name='LaneFollowerNode',
        output='screen',
        parameters=[{'use_sim_time': LaunchConfiguration('use_sim_time')}],
    )
    
    m_horizontal_line_detect_node = Node(
        package='bme_gazebo_sensors_py',
        executable='m_horizontal_line_detect',
        name='M_HorizontalLineDetect',
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
            'camera.image.compressed.jpeg_quality': 75
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
        arguments=['camera/camera_info', 'camera/image/camera_info'],
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
            os.path.join(pkg_bme_gazebo_sensors, 'config', 'ekf.yaml'),
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
            'robot_description': Command(['xacro', ' ', urdf_file_path]),
            'use_sim_time': LaunchConfiguration('use_sim_time')
        }],
        remappings=[('/tf', 'tf'), ('/tf_static', 'tf_static')],
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
    ld.add_action(relay_camera_info_node)
    ld.add_action(ekf_node)
    ld.add_action(robot_state_publisher_node)
    
    ld.add_action(intersection_straight_node)
    ld.add_action(intersection_left_node)
    ld.add_action(gps_waypoint_publisher_node)

    # ld.add_action(move_forward_node)
    
    ld.add_action(m_horizontal_line_detect_node)
    

    return ld
