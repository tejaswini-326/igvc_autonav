#!/usr/bin/env python3
import os

from launch import LaunchDescription 
from launch_ros.actions import Node

def generate_launch_description():
    node1 = Node(
        package='movement',
        executable='parking_pullout',
        output='screen' 
    )

    return LaunchDescription([
        node1
    ])
