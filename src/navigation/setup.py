from setuptools import find_packages, setup

package_name = 'navigation'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', ['launch/path_planner.launch.py']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Abhiyaan',
    maintainer_email= 'abhiyaan@smail.iitm.ac.in',
    description='Package for navigation',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
        'goal_pub= navigation.goal_publisher:main', 
        'costmap= navigation.costmap:main', 
        'path_planner= navigation.path_planning:main', 
        'curve_fit= navigation.lane_curvefitting:main', 
        'controller = navigation.controller:main',
        'downsampling = navigation.downsampling:main',
        'back_downsampling = navigation.back_downsampling:main'
        ],
    },
)