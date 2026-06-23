# Intelligent Ground Vehicle Competition (IGVC) Software Stack

An end-to-end autonomous navigation stack built on **ROS 2** and simulated in **Gazebo**. This repository contains the full autonomy pipeline for a differential drive robot designed to navigate outdoor track environments while avoiding obstacles and staying within designated lane boundaries.

---

## What's in this repository:

* **`ekf/`**: Handles state estimation and sensor fusion.
* **`igvc/`**: Contains the URDF robot model, Gazebo world configurations, and main configuration files.
* **`movement/`**: Houses the path tracking controller and motion execution nodes.
* **`navigation/`**: Implements global path planning algorithms.
* **`object_detection/`**: Manages perception, including camera-based lane tracking and LiDAR-based obstacle isolation.
* **`path_planning/`**: Responsible for costmap generation and fusing perception data into an actionable grid map.

---

## Architecture & Package Descriptions

### 1. Robot & Simulation (`igvc`)
* **Robot Configuration:** A differential drive robot modeled using URDF/XACRO, utilizing a standard two-wheel drive setup with caster wheels for stability.
* **Environment Setup:** A simulated Gazebo environment recreating standard IGVC conditions, populated with white track lanes, construction barrels, and various obstacles.

### 2. State Estimation (`ekf`)
* **Sensor Fusion:** Utilizes an Extended Kalman Filter (EKF) to merge high-rate, noisy sensor feeds into a single robust odometry estimate.
* **Fused Inputs:** Comprises wheel odometry (short-term odometry tracking), IMU data (high-frequency orientation/heading changes), GPS coordinates (global position absolute updates to eliminate drift), and LiDAR data.

### 3. Perception (`object_detection`)
* **Obstacle Isolation:** Combines a 2D YOLO object detection model with 3D LiDAR point cloud data. Bounding boxes from the camera are projected into the 3D space to isolate and extract the precise point clouds corresponding strictly to obstacles.
* **Lane Detection:** A computer vision pipeline that isolates track boundaries using color thresholding (HSV filtering) and maps the layout utilizing polynomial curve fitting.

### 4. Costmap Generation (`path_planning`)
* **Data Fusion:** Merges the isolated obstacle point clouds from the perception layer with the detected lane markings.
* **Grid Map:** Generates a unified 2D occupancy costmap that dynamically marks both physical obstacles (barrels) and virtual boundaries (lanes) as impassable obstacles, strictly bounding the searchable space.

### 5. Path Planning (`navigation`)
* **A\* Global Planner:** Implements a custom A* search algorithm operating directly over the generated 2D costmap.
* **Path Optimization:** Computes the shortest, most efficient collision-free path toward global target coordinates while treating lanes as hard walls.

### 6. Controller (`movement`)
* **Modified Pure Pursuit:** A variation of the classic geometric path-tracking algorithm designed to compute steering and velocity commands based on a look-ahead distance.
* **Dynamic Damping:** Features custom damping logic that automatically dials back linear speed and stabilizes angular velocity when navigating tightly curved paths or approaching obstacles.


### Instructions for running the AI model and associated nodes

add this to your bashrc:

```
export GZ_SIM_RESOURCE_PATH=/<path to igvc-sim/igvc-sim/src/igvc/models
```


Running the object detection node on your pc:

1 -  Make a python virtual environment - so that the model can be loaded. -

```
python3 -m venv ~/yolo_env
```

1.5 -  install these pacakges INSIDE THE VENV: FIRST TIME ONLY

```
pip install --upgrade pip

pip install ultralytics opencv-python numpy
```


2 - build your workspace and source it.

<b>DO NOT BUILD IN VENV. IT WILL RUIN YOUR ROS INSTALLATION.</b>




3- in a separate terminal start the virtual environment

```
source ~/yolo_env/bin/activate
```

4- launch the files as you do normally in the initial terminal

5- run the node in the venv terminal

note: SOURCE THE install/setup.bash OF THE WS HERE BEFORE LAUNCHING

ros2 run object_detection object_data_node.py //make sure to include the .py at the end 

6 - once done kill the node and run this to deactivate the venv

```
deactivate
```
