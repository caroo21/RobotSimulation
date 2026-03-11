# Robot Simulation

3D robot arm simulation using PyBullet with RGBD camera sensing and point cloud processing.

## Overview
Simulates a Kuka IIWA robot arm mounted on a table with surrounding objects. 
A virtual camera captures RGB and depth images, which are used to generate 
and filter 3D point clouds of the scene.

## Stack
- **PyBullet** — physics simulation & rendering
- **NumPy** — data processing
- **Python 3.10**

## File Structure
```
├── new_sim.py          # Main simulation entry point
├── sim_minimum.py      # Minimal simulation setup
├── sensors.py          # Camera config & RGBD capture
├── kinematics.py       # Robot kinematics
├── helper_classes.py   # Shared utilities
├── model.py            # Neural network model
├── test.py             # Point cloud & filter tests
```

## Setup

### 1. Create environment
```bash
conda create -n robotsim python=3.10
conda activate robotsim
```

### 2. Install dependencies
```bash
pip install pybullet numpy
pip freeze > requirements.txt
```

### 3. Run the simulation
```bash
python new_sim.py
```
A PyBullet GUI window will open showing the Kuka arm on the table.  
Press `Ctrl+C` to stop.

## How it works
1. `new_sim.py` sets up the scene (plane, table, robot, objects)
2. `sensors.py` captures an RGBD image from a configurable virtual camera
3. `test.py` processes the depth image into a 3D point cloud and filters it
4. The simulation keeps running at 240Hz until manually stopped
