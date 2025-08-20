#!/usr/bin/env python3
import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import (
    LaunchConfiguration,
    PathJoinSubstitution,
    Command,
    TextSubstitution,
)
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    # ---------------------------------------------------------------------
    # Locate packages and set resource paths
    # ---------------------------------------------------------------------
    pkg_bme = get_package_share_directory('igvc')
    pkg_ros_gz = get_package_share_directory('ros_gz_sim')

    # Prepend models & worlds so Gazebo finds them
    default_world = os.path.join(pkg_bme, 'worlds', 'asphalt.world')
    os.environ['GZ_SIM_RESOURCE_PATH'] = (
        pkg_bme + os.pathsep +
        os.path.join(pkg_bme, 'models') + os.pathsep +
        os.environ.get('GZ_SIM_RESOURCE_PATH', '')
    )

    # ---------------------------------------------------------------------
    # Launch arguments
    # ---------------------------------------------------------------------
    world_arg = DeclareLaunchArgument(
        'world',
        default_value=default_world,
        description='Full path to the Ignition world file to load'
    )
    rviz_arg = DeclareLaunchArgument(
        'rviz', default_value='true',
        description='Whether to start RViz'
    )
    rviz_cfg = DeclareLaunchArgument(
        'rviz_config', default_value='rviz.rviz',
        description='RViz config file name in the rviz/ folder'
    )
    model_arg = DeclareLaunchArgument(
        'model', default_value='mogi_bot.urdf',
        description='URDF filename in the urdf/ folder'
    )
    x_arg = DeclareLaunchArgument('x', default_value='2.5', description='Spawn X')
    y_arg = DeclareLaunchArgument('y', default_value='1.5', description='Spawn Y')
    yaw_arg = DeclareLaunchArgument('yaw', default_value='-1.5707', description='Spawn Yaw')
    sim_time = DeclareLaunchArgument('use_sim_time', default_value='True')

    # URDF path substitution
    urdf_path = PathJoinSubstitution([pkg_bme, 'urdf', LaunchConfiguration('model')])

    # ---------------------------------------------------------------------
    # Include the Gazebo server + client
    # ---------------------------------------------------------------------
    gz_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_ros_gz, 'launch', 'gz_sim.launch.py')
        ),
        launch_arguments={
            'gz_args': [
                LaunchConfiguration('world'),
                TextSubstitution(text=' -r -v 4')
            ],
            'on_exit_shutdown': 'true'
        }.items()
    )

    # ---------------------------------------------------------------------
    # Nodes
    # ---------------------------------------------------------------------
    rviz_node = Node(
        package='rviz2', executable='rviz2',
        arguments=['-d', PathJoinSubstitution([pkg_bme, 'rviz', LaunchConfiguration('rviz_config')])],
        condition=IfCondition(LaunchConfiguration('rviz')),
        parameters=[{'use_sim_time': LaunchConfiguration('use_sim_time')}]
    )

    spawn_node = Node(
        package='ros_gz_sim', executable='create',
        arguments=[
            '-name', 'mogi_bot',
            '-topic', 'robot_description',
            '-x', LaunchConfiguration('x'),
            '-y', LaunchConfiguration('y'),
            '-z', '0.5',
            '-Y', LaunchConfiguration('yaw'),
        ],
        output='screen',
        parameters=[{'use_sim_time': LaunchConfiguration('use_sim_time')}]
    )

    bridge_node = Node(
        package='ros_gz_bridge', executable='parameter_bridge',
        arguments=[
            '/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock',
            '/cmd_vel@geometry_msgs/msg/Twist@gz.msgs.Twist',
            '/odom@nav_msgs/msg/Odometry@gz.msgs.Odometry',
            '/joint_states@sensor_msgs/msg/JointState@gz.msgs.Model',
            '/camera/camera_info@sensor_msgs/msg/CameraInfo@gz.msgs.CameraInfo',
            '/imu@sensor_msgs/msg/Imu@gz.msgs.IMU',
            '/navsat@sensor_msgs/msg/NavSatFix@gz.msgs.NavSat',
            '/scan@sensor_msgs/msg/LaserScan@gz.msgs.LaserScan',
            '/scan/points@sensor_msgs/msg/PointCloud2@gz.msgs.PointCloudPacked',
        ],
        output='screen',
        parameters=[{'use_sim_time': LaunchConfiguration('use_sim_time')}]
    )

    image_bridge = Node(
        package='ros_gz_image', executable='image_bridge',
        arguments=['/camera/image'],
        output='screen',
        parameters=[{
            'use_sim_time': LaunchConfiguration('use_sim_time'),
            'camera.image.compressed.jpeg_quality': 75
        }]
    )

    relay_camera = Node(
        package='topic_tools', executable='relay',
        name='relay_camera_info',
        arguments=['camera/camera_info', 'camera/image/camera_info'],
        output='screen',
        parameters=[{'use_sim_time': LaunchConfiguration('use_sim_time')}]
    )

    ekf = Node(
        package='robot_localization', executable='ekf_node',
        name='ekf_filter_node',
        output='screen',
        parameters=[
            os.path.join(pkg_bme, 'config', 'ekf.yaml'),
            {'use_sim_time': LaunchConfiguration('use_sim_time')}
        ]
    )

    # traj_server = Node(
    #     package='mogi_trajectory_server', executable='mogi_trajectory_server',
    #     name='mogi_trajectory_server'
    # )

    # traj_odom = Node(
    #     package='mogi_trajectory_server',
    #     executable='mogi_trajectory_server_topic_based',
    #     name='mogi_trajectory_server_odom_topic',
    #     parameters=[
    #         {'trajectory_topic': 'trajectory_raw'},
    #         {'odometry_topic': 'odom'}
    #     ]
    # )

    rsp = Node(
        package='robot_state_publisher', executable='robot_state_publisher',
        output='screen',
        parameters=[{
            'robot_description': Command(['xacro ', urdf_path]),
            'use_sim_time': LaunchConfiguration('use_sim_time')
        }],
        remappings=[('/tf', 'tf'), ('/tf_static', 'tf_static')]
    )

    # ---------------------------------------------------------------------
    # Assemble
    # ---------------------------------------------------------------------
    ld = LaunchDescription([
        world_arg, rviz_arg, rviz_cfg, model_arg,
        x_arg, y_arg, yaw_arg, sim_time,
        gz_launch,
        rviz_node,
        spawn_node,
        bridge_node,
        image_bridge,
        relay_camera,
        ekf,
        traj_server,
        traj_odom,
        rsp,
    ])
    return ld
