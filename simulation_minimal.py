import time
import pybullet as p
import pybullet_data

# 1) Physikserver im Direktmodus (ohne GUI) starten
client = p.connect(p.GUI)

# 2) Standard-Suchpfade (URDFs etc.)
p.setAdditionalSearchPath(pybullet_data.getDataPath())

# 3) Schwerkraft und Boden
p.setGravity(0, 0, -9.81)
plane_id = p.loadURDF("plane.urdf")
robot_id = p.loadURDF("kuka_iiwa/model.urdf", basePosition=[0, 0, 0.52])

# 4) Einfache Box als “Roboter”
box_start_pos = [0, 0, 0.5]
box_start_orn = p.getQuaternionFromEuler([0, 0, 0])
box_collision = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.1, 0.1, 0.1])
box_visual = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.1, 0.1, 0.1], rgbaColor=[0.2, 0.6, 0.9, 1])
box_id = p.createMultiBody(baseMass=1.0, baseCollisionShapeIndex=box_collision,
                           baseVisualShapeIndex=box_visual,
                           basePosition=box_start_pos, baseOrientation=box_start_orn)

# 5) Simulationsschritte
for i in range(240):  # ~1 Sekunde bei 240 Hz
    p.stepSimulation()
    time.sleep(1.0/240.0)


