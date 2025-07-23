IGVC Sim

add this to your bashrc:

export GZ_SIM_RESOURCE_PATH=/<path to igvc-sim/igvc-sim/src/igvc/models


Running the object detection node on your pc:

1 -  Make a python virtual environment - so that the model can be loaded. -

python3 -m venv ~/yolo_env

1.5 -  install these pacakges INSIDE THE VENV: FIRST TIME ONLY

pip install --upgrade pip

pip install ultralytics opencv-python numpy


2 - build your workspace

<b>DO NOT BUILD IN VENV. IT WILL RUIN YOUR ROS INSTALLATION.</b>

, source it.



3- in a separate terminal start the virtual environment

source ~/yolo_env/bin/activate

4- launch the files as you do normally in the initial terminal

5- run the node in the venv terminal

note: SOURCE THE install/setup.bash OF THE WS HERE BEFORE LAUNCHING

ros2 run object_detection object_data_node.py //make sure to include the .py at the end 

6 - once done kill the node and run this to deactivate the venv

deactivate
