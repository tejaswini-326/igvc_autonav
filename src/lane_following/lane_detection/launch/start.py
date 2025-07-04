import os
import yaml
from pathlib import Path
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import Command, LaunchConfiguration
from launch_ros.actions import Node
import launch_ros.descriptions

def log(msg):
    print("[lane_detection/start.py]:" + msg)

def generate_launch_description():
    pkg_name = 'lane_detection'
    pkg_share = get_package_share_directory(pkg_name)
    try:
        with open(str(Path.home()) + '/.config/config_igvc_ui/config.yaml', 'r') as file:
            data = yaml.safe_load(file)
            bt_low = data['parameters']['lane_detection']['low']
            bt_high = data['parameters']['lane_detection']['high']
            z_thresh = data['parameters']['lane_detection']['z']
            min_pts = int(data['parameters']['lane_detection']['min_pts'])
            eps = data['parameters']['lane_detection']['eps']
            yellow_r_min = data['parameters']['lane_detection']['yellow_r_min']
            yellow_r_max = data['parameters']['lane_detection']['yellow_r_max']
            yellow_g_min = data['parameters']['lane_detection']['yellow_g_min']
            yellow_g_max = data['parameters']['lane_detection']['yellow_g_max']
            yellow_b_min = data['parameters']['lane_detection']['yellow_b_min']
            yellow_b_max = data['parameters']['lane_detection']['yellow_b_max']
            log(f"{bt_low}, {bt_high}")
    except Exception as e:
        log("did you specify all threshold params?")
        print(e)
        bt_low = 215.0
        bt_high = 245.0
        z_thresh = 0.1
        min_pts = 6
        eps = 0.1
        yellow_r_min = 82.022
        yellow_r_max = 255.000
        yellow_g_min = 95.828
        yellow_g_max = 255.000
        yellow_b_min = 0.000
        yellow_b_max = 102.325
            

    pointcloud_filter = Node(
        package=pkg_name,
        executable='filter',
        name='pc_thrsh',
        parameters=[{'use_sim_time': LaunchConfiguration('use_sim_time')},
                    {'bt_low': bt_low},
                    {'bt_high': bt_high},
                    {'z_thresh': z_thresh}, # night = 240, 255
                    {'yellow_r_min': yellow_r_min},
                    {'yellow_r_max': yellow_r_max},
                    {'yellow_g_min': yellow_g_min}, # night = 240, 255
                    {'yellow_g_max': yellow_g_max},
                    {'yellow_b_min': yellow_b_min},
                    {'yellow_b_max': yellow_b_max}], # night = 240, 255
        output='screen'
    )

    pointcloud_cluster = Node(
        package=pkg_name,
        executable='cluster',
        name='pc_clstr',
        parameters=[{'use_sim_time': LaunchConfiguration('use_sim_time')},
                    {'min_pts': min_pts},
                    {'eps': eps}],
        output='screen'
    )

    return LaunchDescription(
            [

                DeclareLaunchArgument(
                    'use_sim_time',
                    default_value='True',
                    description='simulation or real time'),
                # DeclareLaunchArgument(
                #     'eps',
                #     default_value=0.02,
                #     description='distance for neighbors'),
                # DeclareLaunchArgument(
                #     'min_pts',
                #     default_value=5,
                #     description='frontier points'),
                pointcloud_cluster,
                pointcloud_filter
            ]
        )
