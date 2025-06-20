Running the object detection node on your pc:
0 - download the model and change its path in object_data/src/object_data_node.py
1 -  Make a python virtual environment - so that the model can be loaded. -
python3 -m venv ~/yolo_env 
2 - build your workspace, source it.

DO NOT BUILD IN VENV. IT WILL RUIN YOUR ROS INSTALLATION.

3- in a separate terminal start the virtual environment
source ~/yolo_env/bin/activate
4- launch the files as you do normally in the initial terminal

5- run the node in the venv terminal
ros2 run object_data object_data_node.py //make sure to include the .py at the end 
6 - once done kill the node and run this to deactivate the venv
deactivate
