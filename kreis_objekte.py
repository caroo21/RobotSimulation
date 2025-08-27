import pybullet as p
import pybullet_data
import numpy as np
import time
import math
from kinematics import RobotKinematics

#starten
physicsClient = p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0,0,-9.81)

#Boden laden
planeId = p.loadURDF("plane.urdf")

#einfacher Tisch
table_collision = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.5, 0.5, 0.02])
table_visual = p.createVisualShape(p.GEOM_BOX, halfExtents = [0.5,0.5,0.02], rgbaColor = [0.7,0.4,0.2,1])
table_id = p.createMultiBody(baseMass=0, baseCollisionShapeIndex = table_collision, 
                             baseVisualShapeIndex=table_visual,
                             basePosition=[0,0,0.5])

robot_id = p.loadURDF("kuka_iiwa/model.urdf", basePosition=[0, 0, 0.52])

# Einige Test-Objekte hinzufügen
# Würfel
cube_id = p.loadURDF("cube_small.urdf", basePosition=[0.3, 0.0, 0.6])

# Zylinder erstellen
cylinder_collision = p.createCollisionShape(p.GEOM_CYLINDER, radius=0.03, height=0.1)
cylinder_visual = p.createVisualShape(p.GEOM_CYLINDER, radius=0.03, length=0.1, rgbaColor=[0, 0, 1, 1])
cylinder_id = p.createMultiBody(baseMass=0.1,
                               baseCollisionShapeIndex=cylinder_collision,
                               baseVisualShapeIndex=cylinder_visual,
                               basePosition=[0.2, -0.2, 0.55])

# Kamera-Position für bessere Sicht einstellen
p.resetDebugVisualizerCamera(cameraDistance=1.5, 
                           cameraYaw=45, 
                           cameraPitch=-30, 
                           cameraTargetPosition=[0, 0, 0.5])

# Instantiate kinematics
kinematics = RobotKinematics(robot_id)

print("\n🔄 Demo 1: Kreisbewegung")
center = [0.4, 0.0, 0.8]
radius = 0.15

for i in range(100):
    angle = i * 0.1
    x = center[0] + radius * math.cos(angle)
    y = center[1] + radius * math.sin(angle)
    z = center[2]
    target_pos = [x, y, z]

    # Keep a modest number of steps per move for smooth path, small sleep for speed
    ok = kinematics.move_to_position(target_pos, steps=20, sleep=0.005,
                            abort_on_collision=True,
                            ignore_bodies={planeId, table_id})
    print(f"Schritt {i}: Position [{x:.2f}, {y:.2f}, {z:.2f}]")
    if not ok:
        print("Gesamtablauf stoppen")
        break

input("Drücke Enter zum Fortfahren...")