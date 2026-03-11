import pybullet as p
import pybullet_data
import numpy as np
import time
import math
from kinematics import RobotKinematics 
from helper_classes import PickPlaceExpert, TrajectoryCollector, TrainedPolicyController


def setup_simulation():
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

    robot_id = p.loadURDF("kuka_iiwa/model.urdf", basePosition=[-0.2, 0, 0.52])

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
    
    return physicsClient, planeId, table_id, robot_id, cube_id, cylinder_id



def test_policy(robot_id, cube_id, cylinder_id, planeId, table_id):
    # Kinematics & Policy laden
    kinematics = RobotKinematics(robot_id, ignore_bodies={planeId, table_id})

    policy = TrainedPolicyController(
            model_path='pickplace_policy_tensorflow.h5',
            robot_kinematics=kinematics,
            cube_id=cube_id,
            cylinder_id=cylinder_id
        )
        
    print("Teste trainierte Policy an NEUER Position!")
    cube_pos, _ = p.getBasePositionAndOrientation(cube_id)
    cylinder_pos, _ = p.getBasePositionAndOrientation(cylinder_id)
    print(f"Würfel: [{cube_pos[0]:.2f}, {cube_pos[1]:.2f}, {cube_pos[2]:.2f}]")
    print(f"Zylinder: [{cylinder_pos[0]:.2f}, {cylinder_pos[1]:.2f}, {cylinder_pos[2]:.2f}]")
        
    # Policy ausführen
    policy.execute_policy(max_steps=50)


def collect_data(num_trajectories, robot_id, cube_id, cylinder_id, planeId, table_id):
    # Kinematics & Expert & Datensammler
    kinematics = RobotKinematics(robot_id, ignore_bodies={planeId, table_id})
    expert = PickPlaceExpert(kinematics, cube_id, cylinder_id, ignore_bodies={planeId, table_id})
    collector = TrajectoryCollector(expert, kinematics, cube_id, cylinder_id)
        
    print("Starte Datensammlung...")
        
    # Datensatz sammeln
    trajectories = collector.collect_dataset(num_trajectories)  # Erstmal nur 20 zum Testen
        
    # Speichern
    collector.save_dataset("kuka_trajectories.pkl")
        
    print("\nDatensatz-Statistiken:")
    print(f"Anzahl Trajektorien: {len(trajectories)}")
    if trajectories:
        avg_length = sum(t['length'] for t in trajectories) / len(trajectories)
        print(f"Durchschnittliche Länge: {avg_length:.1f} Schritte")
        
    print("\n✅ Datensammlung abgeschlossen!")
    


def test_PickPlaceExpert(robot_id, cube_id, cylinder_id, planeId, table_id):
    # Instantiate kinematics
    kinematics = RobotKinematics(robot_id)

    # Expert erstellen
    expert = PickPlaceExpert(
        robot_kinematics=kinematics,
        cube_id=cube_id,
        cylinder_id=cylinder_id,
        ignore_bodies={planeId, table_id}
    )
    trajectory_data = []

    print("\nPick-and-Place Expert Demo")
    print("=" * 40)

    # Expert-Demonstration ausführen
    step = 0
    while not expert.is_complete() and step < 100:  # Sicherheits-Limit
        print(f"\nSchritt {step}")
        print(f"Aktuelle Phase: {expert.get_current_phase()}")
        
        #state = get_current_state()  # Roboter + Objekt Positionen
        action = expert.get_action()
        print(f"Action: {action}")
        
        if action.get('complete'):
            break
            
        step += 1
        time.sleep(0.1)  # Kurze Pause für Visualisierung

    print("\nExpert-Demonstration beendet!")


def test_kinematics(planeId, table_id, robot_id):
    kinematics = RobotKinematics(robot_id)
    print("\nKinematics Test")
    print("=" * 20)
    
    # Zielpositionen definieren
    target_positions = [
        [0.4, 0.0, 0.6],
        [0.3, 0.2, 0.5],
        [0.2, -0.2, 0.55]
    ]
    
    for i, pos in enumerate(target_positions):
        print(f"\nBewege zu Position {i}: {pos}")
        ok = kinematics.move_to_position(pos, steps=50, sleep=0.01,
                                        abort_on_collision=True,
                                        ignore_bodies={planeId, table_id})
        if not ok:
            print("Bewegung abgebrochen wegen Kollision oder Fehler.")
        else:
            print("Position erreicht.")
        time.sleep(1)  # Kurze Pause zwischen den Bewegungen

def test_kreis(robot_id, planeId, table_id):
    kinematics = RobotKinematics(robot_id)
    print("\nDemo 1: Kreisbewegung")
    center = [0.4, 0.0, 0.6]
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

if __name__ == "__main__":
    physicsClient, planeId, table_id, robot_id, cube_id, cylinder_id = setup_simulation()
    collect_data(100, robot_id, cube_id, cylinder_id, planeId, table_id)  # Datensammlung starten
    #test_policy(robot_id, cube_id, cylinder_id, planeId, table_id)  # Trainierte Policy testen
    #test_PickPlaceExpert(robot_id, cube_id, cylinder_id, planeId, table_id)  # Expert testen
    #test_kinematics(planeId, table_id, robot_id)  # Kinematics testen
    #test_kreis(robot_id, planeId, table_id)  # Kreisbewegung testen

    print("\nSimulation läuft... Drücke STRG+C zum Beenden.")
    try:
        while True:
            p.stepSimulation()
            time.sleep(1./240.)
    except KeyboardInterrupt:
        pass

    p.disconnect()