from setuptools import find_packages, setup

package_name = 'movement'

setup(
    name=package_name,
    version='1.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='David Dudas',
    maintainer_email='david.dudas@outlook.com',
    description='Python nodes for simulation of various sensors with Gazebo Harmonic and ROS Jazzy for BME MOGI ROS2 course',
    license='Apache License 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'image_republisher = movement.image_republisher:main',
            'chase_the_ball = movement.chase_the_ball:main',
            'transformed_pointcloud = movement.pointcloud_transform:main',

            'move_forward = movement.move_forward:main',

            'intersection_left = movement.intersection_left:main',
            'intersection_straight = movement.intersection_straight:main',
            'm_horizontal_line_detect = movement.m_horizontal_line_detect:main',

            'gps_waypoint_publisher = movement.gps_waypoint_publisher:main',
            'follow_barrel_and_stop = movement.follow_barrel_and_stop:main',         
        ],
    },
)