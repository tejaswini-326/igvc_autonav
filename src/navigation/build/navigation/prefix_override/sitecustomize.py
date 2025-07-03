import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/home/sinan/ros2_ws/src/igvc-sim/src/navigation/install/navigation'
