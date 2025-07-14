from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    nav_py_package = 'navigation'
    path_cpp_package = 'path_planning'
            
    goal_publisher_node = Node(
        package=path_cpp_package,
        executable='goal_publisher',
        name='goal_publisher',
        parameters=[{'use_sim_time': LaunchConfiguration('use_sim_time')}],
        output='screen'
    )

    costmap_publisher_node = Node(
        package=nav_py_package,
        executable='costmap',
        name='costmap_publisher',
        parameters=[{'use_sim_time': LaunchConfiguration('use_sim_time')}],
        output='screen'
    )

    path_publisher_node = Node(
        package=path_cpp_package,
        executable='path_planner',
        name='path_planner',
        parameters=[{'use_sim_time': LaunchConfiguration('use_sim_time')}],
        output='screen'
    )

    curve_fit_node = Node(
        package=nav_py_package,
        executable='curve_fit',
        name='lane_fit',
        parameters=[{'use_sim_time': LaunchConfiguration('use_sim_time')}],
        output='screen'
    )
    controller_node = Node(
        package=nav_py_package,
        executable='controller',
        name='controller',
        parameters=[{'use_sim_time': LaunchConfiguration('use_sim_time')}],
        output='screen'
    )

    return LaunchDescription(
            [
                DeclareLaunchArgument(
                    'use_sim_time',
                    default_value='True',
                    description='simulation or real time'),
                # goal_publisher_node,
                costmap_publisher_node,
                path_publisher_node,
                curve_fit_node,
                # controller_node
            ]
        )