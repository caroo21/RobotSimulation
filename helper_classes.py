import random
import numpy as np
import pickle
import pybullet as p
import time
import tensorflow as tf

class PickPlaceExpert:
    def __init__(self, robot_kinematics, cube_id, cylinder_id, ignore_bodies=None):
        self.robot = robot_kinematics
        self.cube_id = cube_id
        self.cylinder_id = cylinder_id
        self.ignore_bodies = ignore_bodies or set()
        
        self.phases = [
            'approach_cube',
            'move_to_cylinder', 
            'retreat'
        ]
        
        self.current_phase = 0
        
        # Konfiguration
        self.approach_height = 0.20  # 20cm über Objekt
        self.position_tolerance = 0.03  # 3cm Toleranz
        self.movement_steps = 30
        
    def reset(self):
        self.current_phase = 0
        
    def get_current_phase(self):
        if self.current_phase < len(self.phases):
            return self.phases[self.current_phase]
        return 'complete'
        
    def is_complete(self):
        return self.current_phase >= len(self.phases)
        
    def _get_object_positions(self):
        """Aktuelle Objektpositionen aus PyBullet holen"""
        cube_pos, _ = p.getBasePositionAndOrientation(self.cube_id)
        cylinder_pos, _ = p.getBasePositionAndOrientation(self.cylinder_id)
        return np.array(cube_pos), np.array(cylinder_pos)
        
    def _distance_to_target(self, current_pos, target_pos):
        """Euklidische Distanz"""
        return np.linalg.norm(np.array(current_pos) - np.array(target_pos))
        
    def _approach_cube(self, cube_pos, ee_pos):
        """Phase 1: Zum Würfel bewegen"""
        #high_target = cube_pos + np.array([0, 0, 0.35])

        target_pos = cube_pos + np.array([0, 0, self.approach_height])
        
        # Prüfen ob Ziel erreicht
        distance = self._distance_to_target(ee_pos, target_pos)
        if distance < self.position_tolerance:
            print(f"Phase {self.get_current_phase()} abgeschlossen!")
            self.current_phase += 1
            return {
                'phase_complete': True, 
                'next_phase': self.get_current_phase(),
                'distance': distance
            }
        
        # Bewegung ausführen
        success = self.robot.move_to_position(
            target_pos, 
            steps=self.movement_steps, 
            sleep=0.01,
            abort_on_collision=True,
            ignore_bodies=self.ignore_bodies
        )
        
        return {
            'target_position': target_pos.tolist(),
            'current_position': ee_pos,
            'distance': distance,
            'movement_success': success,
            'phase': 'approach_cube'
        }
    
    def _move_to_cylinder(self, cylinder_pos, ee_pos):
        """Phase 2: Zum Zylinder bewegen"""
        target_pos = cylinder_pos + np.array([0, 0, self.approach_height])
        
        distance = self._distance_to_target(ee_pos, target_pos)
        if distance < self.position_tolerance:
            print(f"Phase {self.get_current_phase()} abgeschlossen!")
            self.current_phase += 1
            return {
                'phase_complete': True, 
                'next_phase': self.get_current_phase(),
                'distance': distance
            }
        
        success = self.robot.move_to_position(
            target_pos, 
            steps=self.movement_steps, 
            sleep=0.01,
            abort_on_collision=True,
            ignore_bodies=self.ignore_bodies
        )
        
        return {
            'target_position': target_pos.tolist(),
            'current_position': ee_pos,
            'distance': distance,
            'movement_success': success,
            'phase': 'move_to_cylinder'
        }
    
    def _retreat(self, cylinder_pos, ee_pos):
        """Phase 3: Zurückziehen"""
        target_pos = cylinder_pos + np.array([0, 0, 0.15])  # 15cm über Zylinder
        
        distance = self._distance_to_target(ee_pos, target_pos)
        if distance < self.position_tolerance:
            print(f"Alle Phasen abgeschlossen!")
            self.current_phase += 1
            return {'complete': True, 'distance': distance}
        
        success = self.robot.move_to_position(
            target_pos, 
            steps=self.movement_steps, 
            sleep=0.01,
            abort_on_collision=True,
            ignore_bodies=self.ignore_bodies
        )
        
        return {
            'target_position': target_pos.tolist(),
            'current_position': ee_pos,
            'distance': distance,
            'movement_success': success,
            'phase': 'retreat'
        }
    
    def get_action(self):
        """
        Hauptfunktion - automatisch mit PyBullet Objekten
        """
        if self.is_complete():
            return {'complete': True}
            
        # Aktuelle Positionen holen
        cube_pos, cylinder_pos = self._get_object_positions()
        ee_pos, _ = self.robot.forward_kinematics()
        ee_pos = np.array(ee_pos)
        
        current_phase = self.get_current_phase()
        
        if current_phase == 'approach_cube':
            return self._approach_cube(cube_pos, ee_pos)
            
        elif current_phase == 'move_to_cylinder':
            return self._move_to_cylinder(cylinder_pos, ee_pos)
            
        elif current_phase == 'retreat':
            return self._retreat(cylinder_pos, ee_pos)
            
        return {'complete': True}


class TrajectoryCollector:
    def __init__(self, expert, robot_kinematics, cube_id, cylinder_id):
        self.expert = expert
        self.robot = robot_kinematics
        self.cube_id = cube_id
        self.cylinder_id = cylinder_id
        self.trajectories = []
        
    def randomize_objects(self):
        """Zufällige Positionen für Würfel und Zylinder"""
        # Würfel: Zufällig auf dem Tisch
        cube_x = random.uniform(0.3, 0.5)
        cube_y = random.uniform(0.3, 0.5)
        cube_z = 0.6  # Auf Tischhöhe
        
        # Zylinder: Woanders auf dem Tisch
        cylinder_x = random.uniform(0.3, 0.5)
        cylinder_y = random.uniform(0.3, 0.5)
        cylinder_z = 0.55
        
        # Mindestabstand zwischen Objekten
        while np.linalg.norm([cube_x - cylinder_x, cube_y - cylinder_y]) < 0.15:
            cylinder_x = random.uniform(0.1, 0.4)
            cylinder_y = random.uniform(-0.3, 0.3)
        
        # Positionen setzen
        p.resetBasePositionAndOrientation(self.cube_id, [cube_x, cube_y, cube_z], [0,0,0,1])
        p.resetBasePositionAndOrientation(self.cylinder_id, [cylinder_x, cylinder_y, cylinder_z], [0,0,0,1])
        
        return [cube_x, cube_y, cube_z], [cylinder_x, cylinder_y, cylinder_z]
    
    def collect_trajectory(self):
        """Eine komplette Trajektorie sammeln"""
        trajectory = []
        
        self.expert.reset()
        step = 0
        
        while not self.expert.is_complete() and step < 200:
            # Aktuellen Zustand erfassen
            state = self.get_current_state()
            
            # Expert-Aktion
            action = self.expert.get_action()

             # Kollision behandeln
            if action.get('collision') or action.get('trajectory_failed'):
                collision_count += 1
                print(f" Kollision #{collision_count} in Schritt {step}")
                
                if collision_count > 3:  # Zu viele Kollisionen
                    print("Trajektorie abgebrochen: Zu viele Kollisionen")
                    return []  # Leere Trajektorie = Fehlschlag
                
                # Neue Objektpositionen versuchen
                if action.get('reposition_objects'):
                    print("Repositioniere Objekte...")
                    self.randomize_objects()
                    self.expert.reset()
                    trajectory = []  # Neu starten
                    step = 0
                    continue
            
            # Speichern
            trajectory.append({
                'state': state,
                'action': action,
                'step': step
            })
            
            if action.get('complete'):
                break
                
            step += 1
            
        return trajectory
    
    def get_current_state(self):
        """Aktuellen Zustand des Systems erfassen"""
        # Roboter-Zustand
        joint_positions = self.robot.get_joint_positions()
        ee_pos, ee_orient = self.robot.forward_kinematics()
        
        # Objekt-Positionen
        cube_pos, _ = p.getBasePositionAndOrientation(self.cube_id)
        cylinder_pos, _ = p.getBasePositionAndOrientation(self.cylinder_id)
        
        return {
            'joint_positions': joint_positions.tolist(),
            'end_effector_pos': list(ee_pos),
            'end_effector_orient': list(ee_orient),
            'cube_position': list(cube_pos),
            'cylinder_position': list(cylinder_pos),
            'timestamp': time.time()
        }
    
    def collect_dataset(self, num_trajectories=100):
        """Kompletten Datensatz sammeln"""
        print(f"Sammle {num_trajectories} Trajektorien...")
        
        successful_trajectories = 0
        
        for i in range(num_trajectories):
            print(f"\nTrajektorie {i+1}/{num_trajectories}")
            
            # Zufällige Objektpositionen
            cube_pos, cylinder_pos = self.randomize_objects()
            print(f"Würfel: {cube_pos}")
            print(f"Zylinder: {cylinder_pos}")
            
            # Trajektorie sammeln
            trajectory = self.collect_trajectory()
            
            if len(trajectory) > 5:  # Mindestlänge
                self.trajectories.append({
                    'trajectory': trajectory,
                    'cube_start': cube_pos,
                    'cylinder_start': cylinder_pos,
                    'success': True,
                    'length': len(trajectory)
                })
                successful_trajectories += 1
                print(f"Erfolgreich! ({len(trajectory)} Schritte)")
            else:
                print("Trajektorie zu kurz")
        
        print(f"\nFertig! {successful_trajectories}/{num_trajectories} erfolgreiche Trajektorien")
        return self.trajectories
    
    def save_dataset(self, filename="trajectories.pkl"):
        """Datensatz speichern"""
        with open(filename, 'wb') as f:
            pickle.dump(self.trajectories, f)
        print(f"Datensatz gespeichert: {filename}")


class TrainedPolicyController:
    def __init__(self, model_path, robot_kinematics, cube_id, cylinder_id):
        # Trainiertes Model laden
        self.model = tf.keras.models.load_model(model_path, compile=False)
        self.robot = robot_kinematics
        self.cube_id = cube_id
        self.cylinder_id = cylinder_id
        
        print(f"Trainiertes Model geladen: {model_path}")
        
    def get_current_state(self):
        """Aktuellen Zustand erfassen (wie beim Training)"""
        # Roboter-Zustand
        joint_positions = self.robot.get_joint_positions()
        ee_pos, ee_orient = self.robot.forward_kinematics()
        
        # Objekt-Positionen
        cube_pos, _ = p.getBasePositionAndOrientation(self.cube_id)
        cylinder_pos, _ = p.getBasePositionAndOrientation(self.cylinder_id)
        
        # Feature-Vektor erstellen (genau wie beim Training!)
        features = []
        features.extend(joint_positions.tolist())  # 7 Werte
        features.extend(list(ee_pos))              # 3 Werte  
        features.extend(list(cube_pos))            # 3 Werte
        features.extend(list(cylinder_pos))        # 3 Werte
        
        return np.array(features, dtype=np.float32).reshape(1, -1)  # Batch dimension
    
    def predict_next_action(self):
        """Model vorhersage für nächste Aktion"""
        state = self.get_current_state()
        predicted_action = self.model.predict(state, verbose=0)[0]  # [x, y, z]
        return predicted_action
    
    def execute_policy(self, max_steps=100):
        """Trainierte Policy ausführen"""
        print("Starte trainierte Policy...")
        
        for step in range(max_steps):
            print(f"\n--- Schritt {step+1} ---")
            
            # Aktuelle Position
            ee_pos, _ = self.robot.forward_kinematics()
            print(f"Aktuelle Position: [{ee_pos[0]:.3f}, {ee_pos[1]:.3f}, {ee_pos[2]:.3f}]")
            
            # Model Vorhersage
            target_pos = self.predict_next_action()
            print(f"Ziel Position:     [{target_pos[0]:.3f}, {target_pos[1]:.3f}, {target_pos[2]:.3f}]")
            
            # Distanz prüfen
            distance = np.linalg.norm(np.array(ee_pos) - target_pos)
            print(f"Distanz zum Ziel:  {distance:.3f}")
            
            # Wenn sehr nah am Ziel, stoppen
            if distance < 0.02:  # 2cm
                print("Ziel erreicht!")
                break
            
            # Bewegung ausführen
            success = self.robot.move_to_position(
                target_pos, 
                steps=20, 
                sleep=0.02,
                abort_on_collision=True
            )
            
            if not success:
                print("Kollision oder Fehler!")
                break
                
            print(f"Bewegung: {'Erfolgreich' if success else 'Fehlgeschlagen'}")
