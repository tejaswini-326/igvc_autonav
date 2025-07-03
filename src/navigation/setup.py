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
    maintainer='tejaswini',
    maintainer_email='tejaswinianbazhagan@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
<<<<<<< HEAD:src/costmap/setup.py
        'costmap_node= costmap.costmap:main',
        'controller_node=costmap.controller:main' 
=======
        'goal_pub= navigation.goal_publisher:main', 
        'costmap= navigation.costmap:main', 
        'path_planner= navigation.path_planning:main', 
        'curve_fit= navigation.lane_curvefitting:main', 
>>>>>>> fe502660a0c05a2ff3549cc021a9e1ee136f51d6:src/navigation/setup.py
        ],
    },
)
