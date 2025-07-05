from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    pkg_name = 'navigation'
            

    costmap_publisher_node = Node(
        package=pkg_name,
        executable='costmap',
        name='costmap_publisher',
        parameters=[{'use_sim_time': LaunchConfiguration('use_sim_time')}],
        output='screen'
    )


    curve_fit_node = Node(
        package=pkg_name,
        executable='curve_fit',
        name='lane_fit',
        parameters=[{'use_sim_time': LaunchConfiguration('use_sim_time')}],
        output='screen'
    )


    return LaunchDescription(
            [
                DeclareLaunchArgument(
                    'use_sim_time',
                    default_value='True',
                    description='simulation or real time'),
                costmap_publisher_node,
                curve_fit_node,
            ]
        )
