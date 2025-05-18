import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, TextSubstitution
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    # ---------------------------------------------------------------------
    # Locate packages
    # ---------------------------------------------------------------------
    pkg_bme = get_package_share_directory('bme_gazebo_sensors')
    media_root = pkg_bme
    pkg_ros_gz = get_package_share_directory('ros_gz_sim')

    # ---------------------------------------------------------------------
    # World argument (defaults to the asphalt world inside this package)
    # ---------------------------------------------------------------------
    default_world_path = os.path.join(pkg_bme, 'worlds', 'asphault.world')
    world_arg = DeclareLaunchArgument(
        'world',
        default_value=default_world_path,
        description='Absolute path to the Ignition world file (.world or .sdf)'
    )

    # ---------------------------------------------------------------------
    # Make sure Gazebo can find all your custom models
    # ---------------------------------------------------------------------
    models_dir = os.path.join(pkg_bme, 'models') 
    # Pre-pend the models directory to GZ_SIM_RESOURCE_PATH for this process
    os.environ['GZ_SIM_RESOURCE_PATH'] = (
        media_root + os.pathsep +
        models_dir + os.pathsep + 
        os.environ.get('GZ_SIM_RESOURCE_PATH', '')
    )

    # ---------------------------------------------------------------------
    # Include the standard ros_gz_sim launcher and pass world + extra args
    # ---------------------------------------------------------------------
    gazebo_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_ros_gz, 'launch', 'gz_sim.launch.py')
        ),
        launch_arguments={
            # Path to world, followed by Gazebo CLI flags
            'gz_args': [
                LaunchConfiguration('world'),
                TextSubstitution(text=' -r -v 4')
            ],
            'on_exit_shutdown': 'true'   # tidy shutdown when Ctrl-C
        }.items()
    )

    # ---------------------------------------------------------------------
    # Assemble description
    # ---------------------------------------------------------------------
    ld = LaunchDescription()
    ld.add_action(world_arg)
    ld.add_action(gazebo_launch)
    return ld
