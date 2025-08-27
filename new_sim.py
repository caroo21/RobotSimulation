import pybullet as p
import pybullet_data
import numpy as np
import time
import math 
from sensors import CameraConfig, get_rgbd_with_config
from test import test_filter_table_once

def setup_scene():
    # Verbindung mit GUI
    cid = p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)

    # Boden
    plane_id = p.loadURDF("plane.urdf")

    # Tisch (einfacher Quader)
    table_collision = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.5, 0.5, 0.02])
    table_visual = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.5, 0.5, 0.02], rgbaColor=[0.7, 0.4, 0.2, 1])
    table_id = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=table_collision,
                                 baseVisualShapeIndex=table_visual,
                                 basePosition=[0, 0, 0.5])

    # Roboter
    robot_id = p.loadURDF("kuka_iiwa/model.urdf", basePosition=[0, 0, 0.52], useFixedBase=True)

    # Ein paar Objekte
    cube_id = p.loadURDF("cube_small.urdf", basePosition=[0.3, 0.3, 0.55])

    cylinder_collision = p.createCollisionShape(p.GEOM_CYLINDER, radius=0.03, height=0.1)
    cylinder_visual = p.createVisualShape(p.GEOM_CYLINDER, radius=0.03, length=0.1, rgbaColor=[0, 0, 1, 1])
    cylinder_id = p.createMultiBody(baseMass=0.1,
                                    baseCollisionShapeIndex=cylinder_collision,
                                    baseVisualShapeIndex=cylinder_visual,
                                    basePosition=[0.2, -0.2, 0.55])

    # GUI-Ansicht (nur für dich, nicht für Sensoren)
    p.resetDebugVisualizerCamera(cameraDistance=1.5,
                                 cameraYaw=45,
                                 cameraPitch=-30,
                                 cameraTargetPosition=[0, 0, 0.5])

    return dict(
        cid=cid,
        plane_id=plane_id,
        table_id=table_id,
        robot_id=robot_id,
        cube_id=cube_id,
        cylinder_id=cylinder_id
    )

def main():
    scene = setup_scene()

    # Simulation kurz laufen lassen (sichtbar in GUI)
    for _ in range(60):
        p.stepSimulation()
        time.sleep(1.0 / 60.0)

    cam_cfg = CameraConfig(eye=(0.4, 0.0, 0.7),      # Näher und niedriger (war: 0.7, 0.0, 1.1)
    target=(0.0, 0.0, 0.52),  # Direkt auf Tischplatte (war: 0.0, 0.0, 0.6)
    fov_deg=75.0,             # Weiterer Blickwinkel (war: 60.0)
    max_range=1.5)
    rgb, depth_m, seg = get_rgbd_with_config(cam_cfg)
    print("Config capture OK:", rgb.shape, depth_m.shape)

    test_filter_table_once()

    # Noch etwas Simulation laufen lassen, damit GUI offen bleibt
    try:
        while True:
            p.stepSimulation()
            time.sleep(1.0 / 240.0)
    except KeyboardInterrupt:
        pass

    p.disconnect()

if __name__ == "__main__":
    main()
