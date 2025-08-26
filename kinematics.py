import pybullet as p
import pybullet_data
import numpy as np
import time
import math

class RobotKinematics:
    def __init__(self, robot_id, ignore_bodies=None):
        self.ignore_bodies = set(ignore_bodies or [])
        self.robot_id  = robot_id
        self.num_joints = p.getNumJoints(robot_id)

        # aktive Gelenke
        self.active_joints = []
        self.joint_limits = []

        for i in range(self.num_joints):
            joint_info = p.getJointInfo(robot_id, i)
            joint_type = joint_info[2]

            #bewegliche elemente, revoulte=0 prismatic=1
            if joint_type in [p.JOINT_REVOLUTE, p.JOINT_PRISMATIC]:
                self.active_joints.append(i)
                lower_limit = joint_info[8]
                upper_limit = joint_info[9]
                self.joint_limits.append((lower_limit, upper_limit))
        print(f"aktive Gelenke: {len(self.active_joints)}")
        print(f"Gelenke IDs: {self.active_joints}")
        # Choose end effector link explicitly for KUKA iiwa (last moving link)
        self.end_effector_id = self.active_joints[-1]

        # Disable default motors (velocity) so we can use position control comfortably
        for j in self.active_joints:
            p.setJointMotorControl2(self.robot_id, j, controlMode=p.VELOCITY_CONTROL, force=0)


    def get_joint_positions(self):
        """aktuelle Gelenkpositionen abrufen"""
        positions = []
        for joint_id in self.active_joints:
            joint_state= p.getJointState(self.robot_id, joint_id)
            positions.append(joint_state[0])

        return np.array(positions)
        
    def set_joint_positions(self, positions):
        """Gelekepositionen setzen"""
        for i, joint_id in enumerate(self.active_joints):
            if i < len(positions):
                p.setJointMotorControl2(self.robot_id, joint_id, 
                                            p.POSITION_CONTROL,
                                            targetPosition = positions[i])
        
    def forward_kinematics(self, joint_positions=None):
        """von Gelenkwinkel zu Bewegung"""
        self.end_effector_id = self.active_joints[-1]
        if joint_positions is not None:
            self.set_joint_positions(joint_positions)
            p.stepSimulation()
        
        link_state= p.getLinkState(self.robot_id, self.end_effector_id)
        position = link_state[0]
        orientation= link_state[1]

        return position, orientation
        
    def inverse_kinematics(self, target_position, target_orientation=None):

        if target_orientation is None:
            joint_positions = p.calculateInverseKinematics(
                    self.robot_id, self.end_effector_id, target_position)
        else:
            joint_positions = p.calculateInverseKinematics(
                    self.robot_id, self.end_effector_id, target_position, target_orientation)
        
        return joint_positions[:len(self.active_joints)]
    
    def _clamp_to_limits(self, q):
        q_clamped = []
        for i, (lo, hi) in enumerate(self.joint_limits):
            # Some URDFs use huge limits; skip clamp if lo>=hi (unbounded)
            if hi > lo:
                q_clamped.append(np.clip(q[i], lo, hi))
            else:
                q_clamped.append(q[i])
        return np.array(q_clamped)
    
    def check_collisions(self, extra_ignore=None):
        """
        True, wenn irgendein Link des Roboters mit einem anderen Body (außer ignore_bodies)
        in Kontakt ist (distance <= 0.0).
        """
        # Standard: Boden und Tisch ignorieren
        ignore = set(self.ignore_bodies)
        if extra_ignore:
            ignore.update(extra_ignore)

        for c in p.getContactPoints(bodyA=self.robot_id):
            bodyB = c[2]      # ID des anderen Körpers
            distance = c[8]   # <= 0.0 bedeutet Kontakt/Penetration
            if bodyB in ignore:
                continue
            if distance <= 0.0:
                return True
        return False


    def get_collision_details(self, extra_ignore=None):
        """
    Liefert detaillierte Infos zu relevanten Kollisionen (ohne ignore_bodies).
    Jede Kollision enthält Link-IDs, bodyB, Abstand, normale Kraft, Kontaktpositionen.
    """
        ignore = set(self.ignore_bodies)
        if extra_ignore:
            ignore.update(extra_ignore)

        collisions = []
        for c in p.getContactPoints(bodyA=self.robot_id):
            bodyB = c[2]
            distance = c[8]
            if bodyB in ignore:
                continue
            if distance <= 0.0:
                collisions.append({
                "linkA": c[3],                 # Roboter-Link-Index
                "linkB": c[4],                 # Link-Index von bodyB (oder -1 für Base)
                "bodyB": bodyB,                # Body-ID des anderen Körpers
                "distance": distance,          # <= 0.0 = Kontakt
                "normal_force": c[9],          # geschätzte Kontaktkraft
                "positionOnA": c[5],           # Weltkoordinate Kontaktpunkt auf A
                "positionOnB": c[6],           # Weltkoordinate Kontaktpunkt auf B
                "contactNormalOnB": c[7],      # Normalenrichtung auf B
                })
        return collisions

                
    def move_to_position(self, target_position, target_orientation = None, steps=100, sleep=0.01, abort_on_collision=True, ignore_bodies=None):
        """smoothe Bewegung zur Zielposition"""
            #ik berechnen
        target_joints = self.inverse_kinematics(target_position, target_orientation)
        target_joints = self._clamp_to_limits(target_joints)
        current_joints = self.get_joint_positions()
        # Interpolate in joint space
        for s in range(1, steps + 1):
            alpha = s / steps
            q = (1 - alpha) * current_joints + alpha * target_joints
            self.set_joint_positions(q)
            p.stepSimulation()
            if abort_on_collision and self.check_collisions(extra_ignore=ignore_bodies):
                print("Bewegung abgebrochen: Kollision erkannt (Boden/Tisch ignoriert).")
                return False
            if sleep:
                time.sleep(sleep)
        return True