import pybullet as p
import pybullet_data
import time
import numpy as np
from robot_descriptions.loaders.pybullet import load_robot_description

class RobotArm6DOF:
    def __init__(self, robot_name=None, urdf_path=None, base_position=[0, 0, 0], base_orientation=[0, 0, 0, 1], physicsClientId=None):
        """
        Initializes the RobotArm6DOF class.

        Args:
            robot_name (str): Name of the robot description to load (preferred method).
            urdf_path (str): Path to the URDF file (alternative if robot_name not available).
            base_position (list): Base position [x, y, z].
            base_orientation (list): Base orientation as a quaternion [x, y, z, w].
        """
        # self.physicsClient = p.connect(p.GUI)  # or p.DIRECT for non-graphical version
        # p.setGravity(0, 0, -9.81)
        # p.setAdditionalSearchPath(pybullet_data.getDataPath())
        if physicsClientId is None:
            # If no client provided, create one (default to GUI, but can change)
            # For the data generation script, we'll provide a DIRECT client
            self.physicsClient = p.connect(p.GUI)
            print("Created new PyBullet GUI client.")
        else:
            # Use the provided client ID
            self.physicsClient = physicsClientId
            print(f"Using existing PyBullet client: {self.physicsClient}")

        # --- IMPORTANT: Only set gravity and load plane if WE created the client ---
        if physicsClientId is None:
            p.setGravity(0, 0, -9.81)
            p.setAdditionalSearchPath(pybullet_data.getDataPath())
            self.planeId = p.loadURDF("plane.urdf")
        else:
             self.planeId = -1 # Indicate plane wasn't loaded by this instance

        # Load the plane (optional, but good for visualization)
        self.planeId = p.loadURDF("plane.urdf")

        # Load the robot arm
        robot_loaded = False
        if robot_name is not None:
            try:
                # Load robot using robot_descriptions and let it handle base pose/fixation
                self.robotId = load_robot_description(robot_name, basePosition=base_position, baseOrientation=base_orientation, useFixedBase=True)
                robot_loaded = True
            except Exception as e:
                print(f"Failed to load robot description '{robot_name}': {e}")
                if urdf_path is None:
                    raise ValueError("Could not load robot description and no URDF path provided")
                print("Falling back to URDF path")
                # Fallback uses p.loadURDF directly
        
        if not robot_loaded:
            if urdf_path is not None:
                self.robotId = p.loadURDF(urdf_path, basePosition=base_position, baseOrientation=base_orientation, useFixedBase=True)
            else:
                raise ValueError("Either robot_name or urdf_path must be provided and loadable")

        # --- Identify Movable Joints ---
        self.num_total_joints = p.getNumJoints(self.robotId)
        self.movable_joint_indices = []
        self.joint_info = []
        self.joint_lower_limits = []
        self.joint_upper_limits = []
        self.joint_ranges = []

        for i in range(self.num_total_joints):
            info = p.getJointInfo(self.robotId, i)
            # Check if the joint is revolute or prismatic (i.e., not fixed)
            if info[2] == p.JOINT_REVOLUTE or info[2] == p.JOINT_PRISMATIC:
                self.movable_joint_indices.append(i)
                self.joint_info.append(info) # Store info only for movable joints
                self.joint_lower_limits.append(info[8])
                self.joint_upper_limits.append(info[9])
                self.joint_ranges.append(info[9] - info[8])

        self.num_movable_joints = len(self.movable_joint_indices)

        # Determine End Effector Link Index
        if not self.movable_joint_indices:
             raise Exception("Robot has no movable joints!")
        self.end_effector_link_index = self.movable_joint_indices[-1] 
        print(f"Identified {self.num_movable_joints} movable joints with indices: {self.movable_joint_indices}")
        print(f"Using Link Index {self.end_effector_link_index} as End Effector.")

        # Disable default velocity control ONLY for movable joints
        p.setJointMotorControlArray(self.robotId, self.movable_joint_indices, p.VELOCITY_CONTROL, forces=[0] * self.num_movable_joints)

    def reset_robot(self, joint_angles=None):
        """
        Resets the robot's movable joints to a specified configuration or the home position.

        Args:
            joint_angles (list, optional): List of joint angles for movable joints.
                                           If None, resets to zeros.
        """
        if joint_angles is None:
            target_positions = [0.0] * self.num_movable_joints
        else:
            assert len(joint_angles) == self.num_movable_joints, \
                f"Incorrect number of joint angles provided. Expected {self.num_movable_joints}, got {len(joint_angles)}."
            target_positions = joint_angles

        for i, joint_index in enumerate(self.movable_joint_indices):
            p.resetJointState(self.robotId, joint_index, targetValue=target_positions[i])
            # Optionally apply position control to hold the reset state
            # p.setJointMotorControl2(self.robotId, joint_index, p.POSITION_CONTROL, targetPosition=target_positions[i])


    def get_joint_states(self):
        """
        Gets the current positions, velocities, applied torques for movable joints.

        Returns:
            tuple: (joint_positions, joint_velocities, joint_torques) for movable joints.
        """
        joint_states = p.getJointStates(self.robotId, self.movable_joint_indices)
        joint_positions = [state[0] for state in joint_states]
        joint_velocities = [state[1] for state in joint_states]
        joint_torques = [state[3] for state in joint_states]  # Applied Joint Effort
        return joint_positions, joint_velocities, joint_torques

    def forward_kinematics(self, joint_angles):
        """
        Calculates the end-effector pose (position and orientation) using forward kinematics.
        Temporarily resets ONLY the movable joints to calculate FK.

        Args:
            joint_angles (list): List of joint angles for the movable joints.

        Returns:
            tuple: (position, orientation) - End-effector world position (x, y, z) and orientation (quaternion x, y, z, w).
                   Returns None if the number of joint angles is incorrect.
        """
        if len(joint_angles) != self.num_movable_joints:
            print(f"Error: Incorrect number of joint angles for FK. Expected {self.num_movable_joints}, got {len(joint_angles)}.")
            return None # Return None explicitly on error

        for i, joint_index in enumerate(self.movable_joint_indices):
            # Use resetJointState for immediate effect, suitable for pure FK calculation
            p.resetJointState(self.robotId, joint_index, joint_angles[i])

        # --- Use the determined end_effector_link_index ---
        link_state = p.getLinkState(self.robotId, self.end_effector_link_index, computeForwardKinematics=True)

        # Restore previous state if needed
        # for i, joint_index in enumerate(self.movable_joint_indices):
        #     p.resetJointState(self.robotId, joint_index, current_positions[i])

        if link_state is None:
             print(f"Error: getLinkState failed for link index {self.end_effector_link_index}")
             return None # Return None if getLinkState fails

        position = link_state[0]  # World position is index 0 in modern PyBullet (was 4)
        orientation = link_state[1]  # World orientation is index 1 (was 5)
        
        # Check if results are valid (sometimes PyBullet might return unexpected values if joints are invalid)
        if position is None or orientation is None:
             print("Warning: FK calculation resulted in None for position or orientation.")
             return None

        return position, orientation


    def inverse_kinematics(self, target_position, target_orientation=None, solver=0, max_iterations=100, tolerance=1e-4):
        """
        Calculates the joint angles for movable joints required to reach a target end-effector pose.

        Args:
            target_position (list): Target end-effector position [x, y, z].
            target_orientation (list, optional): Target end-effector orientation as a quaternion [x, y, z, w].
                If None, the solver might ignore orientation or use a default. Defaults to None.
            solver (int): IK solver type (PyBullet specifics may vary). 0 is often DLS.
            max_iterations (int): Maximum iterations for the solver.
            tolerance (float): Residual threshold for the solution.

        Returns:
            list: List of joint angles for movable joints, or None if no solution is found.
        """
        kwargs = {
            "bodyUniqueId": self.robotId,
            "endEffectorLinkIndex": self.end_effector_link_index,
            "targetPosition": target_position,
            "maxNumIterations": max_iterations,
            "residualThreshold": tolerance,
        }

        # Add orientation if provided
        if target_orientation is not None:
            kwargs["targetOrientation"] = target_orientation
            
        # Add solver-specific arguments (adjust based on PyBullet version/needs)
        # For solvers that use joint limits/ranges/rest poses:
        if solver != 0: # Assuming solver 0 (DLS) doesn't use these directly in the basic call
            kwargs["lowerLimits"] = self.joint_lower_limits
            kwargs["upperLimits"] = self.joint_upper_limits
            kwargs["jointRanges"] = self.joint_ranges
            # Provide a rest pose matching the number of movable joints
            kwargs["restPoses"] = [0.0] * self.num_movable_joints 

        try:
            joint_angles_raw = p.calculateInverseKinematics(**kwargs)
            if len(joint_angles_raw) >= self.num_movable_joints:
                 # Take the first 'num_movable_joints' angles, assuming they correspond
                 joint_angles = list(joint_angles_raw[:self.num_movable_joints]) 
                 # Sanity check length
                 if len(joint_angles) == self.num_movable_joints:
                     return joint_angles
                 else:
                      print(f"Warning: IK returned {len(joint_angles_raw)} values, expected at least {self.num_movable_joints}.")
                      return None # Or handle differently
            else:
                print(f"Warning: IK returned fewer values ({len(joint_angles_raw)}) than expected movable joints ({self.num_movable_joints}).")
                return None

        except Exception as e:
            print(f"Inverse kinematics failed: {e}")
            return None


    def forward_dynamics(self, joint_torques):
        """
        Applies joint torques to the movable joints and simulates one step.

        Args:
            joint_torques (list): List of joint torques for movable joints.

        Returns: None
        """
        if len(joint_torques) != self.num_movable_joints:
            print(f"Error: Incorrect number of joint torques. Expected {self.num_movable_joints}, got {len(joint_torques)}.")
            return

        p.setJointMotorControlArray(self.robotId, 
                                    self.movable_joint_indices, 
                                    p.TORQUE_CONTROL, 
                                    forces=joint_torques)
        p.stepSimulation()


    def inverse_dynamics(self, desired_joint_positions, desired_joint_velocities, desired_joint_accelerations):
        """
        Calculates the joint torques for movable joints required to achieve desired states.
        Note: PyBullet's inverse dynamics might require states for all bodies/joints,
        or it might operate only on the provided subset. We assume it works with the movable joints here.
        Requires careful testing.

        Args:
            desired_joint_positions (list): Desired positions for movable joints.
            desired_joint_velocities (list): Desired velocities for movable joints.
            desired_joint_accelerations (list): Desired accelerations for movable joints.

        Returns:
            list: List of calculated joint torques for movable joints, or None if input lengths are incorrect.
        """
        num = self.num_movable_joints
        if not (len(desired_joint_positions) == num and len(desired_joint_velocities) == num and len(desired_joint_accelerations) == num):
            print(f"Error: Incorrect input lengths for inverse dynamics. Expected {num} for each.")
            return None
        try:
            torques = p.calculateInverseDynamics(self.robotId,
                                                objPositions=desired_joint_positions, # Assumes these correspond to the DoFs PyBullet uses
                                                objVelocities=desired_joint_velocities,
                                                objAccelerations=desired_joint_accelerations)
            
            # Check if the returned torque vector has the expected length
            if torques is not None and len(torques) >= self.num_movable_joints:
                 # Assume the first num_movable_joints correspond to our controlled joints
                 return list(torques[:self.num_movable_joints])
            else:
                 print(f"Warning: Inverse dynamics returned {torques} (length {len(torques) if torques else 0}). Expected {self.num_movable_joints}.")
                 return None # Return None if result is unexpected

        except Exception as e:
            print(f"Inverse dynamics calculation failed: {e}")
            return None

    def move_to_target(self, target_position, target_orientation=None, max_steps=500, control_mode='position', position_gain=0.05, velocity_gain = 1.0, max_force=50):
        """
        Moves the robot's end-effector to a target pose using IK and joint control.

        Args:
            target_position (list): Target end-effector position [x, y, z].
            target_orientation (list, optional): Target end-effector orientation [x, y, z, w]. Defaults to None.
            max_steps (int): Maximum simulation steps to attempt reaching the target.
            control_mode (str): 'position' for simple position control, 'torque' for basic torque-based PD control.
            position_gain (float): Proportional gain for position control or PD control.
            velocity_gain (float): Derivative gain for PD control (torque mode). Not used in 'position' mode directly by setJointMotorControlArray.
            max_force (float): Maximum force/torque applied per joint in position control.
        """
        print(f"Attempting to move to Position: {target_position}, Orientation: {target_orientation}")
        target_joint_angles = self.inverse_kinematics(target_position, target_orientation, solver=1) # Use solver 1 for limits potentially

        if target_joint_angles is None:
            print("No IK solution found. Cannot move.")
            return False # Indicate failure

        if len(target_joint_angles) != self.num_movable_joints:
             print(f"Error: IK returned {len(target_joint_angles)} angles, but expected {self.num_movable_joints}.")
             return False


        print(f"IK Solution (movable joints): {np.round(target_joint_angles, 3)}")

        if control_mode == 'position':
            # Use POSITION_CONTROL with gains specified
            p.setJointMotorControlArray(
                self.robotId,
                self.movable_joint_indices,
                p.POSITION_CONTROL,
                targetPositions=target_joint_angles,
                # PyBullet's POSITION_CONTROL has implicit velocity control.
                # Gains help tune the response. Forces provide limits.
                positionGains=[position_gain] * self.num_movable_joints, 
                velocityGains=[velocity_gain] * self.num_movable_joints, 
                forces=[max_force] * self.num_movable_joints 
            )

            for step in range(max_steps):
                p.stepSimulation()
                time.sleep(1. / 240.)

                # Check for convergence (using FK based on current movable joint states)
                current_joint_positions, _, _ = self.get_joint_states()
                fk_result = self.forward_kinematics(current_joint_positions)
                
                if fk_result:
                    current_pos, current_orn = fk_result
                    pos_error = np.linalg.norm(np.array(current_pos) - np.array(target_position))
                    # Orientation error is trickier (e.g., quaternion distance) - optional
                    # orn_error = ... 
                    
                    if pos_error < 1e-3: # Position tolerance
                        print(f"Reached target position within tolerance after {step} steps.")
                        return True # Indicate success
                else:
                     # If FK fails during movement, something is wrong
                     print("Warning: FK failed during move_to_target loop.")
                     # Continue trying or break? Let's continue for now.

            print("Failed to reach target position within max_steps.")
            return False # Indicate failure


        elif control_mode == 'torque':
            # Basic PD control implementation
            kp = position_gain # Reuse gain names for simplicity
            kd = velocity_gain

            print("Starting torque control loop...")
            for step in range(max_steps):
                current_joint_positions, current_joint_velocities, _ = self.get_joint_states()

                position_errors = [target - current for target, current in zip(target_joint_angles, current_joint_positions)]
                # Target velocity is zero for reaching a static pose
                velocity_errors = [0 - vel for vel in current_joint_velocities] 

                # Calculate desired torques: Base torque (gravity/Coriolis) + PD control torque
                # Getting accurate base torques via inverse dynamics in a loop is tricky.
                # Often simplified to just PD control + maybe gravity compensation if calculated separately.
                # Let's try a simple PD controller first.

                control_torques = [
                    kp * pos_err + kd * vel_err
                    for pos_err, vel_err in zip(position_errors, velocity_errors)
                ]
                
                # Apply clamping to torques (important for stability)
                control_torques = np.clip(control_torques, -max_force, max_force).tolist()


                # Apply torques using forward dynamics
                self.forward_dynamics(control_torques) # Also steps simulation
                time.sleep(1./240)

                # Check convergence (similar to position control)
                # Re-calculate FK based on the state *after* the step
                current_joint_positions_after, _, _ = self.get_joint_states()
                fk_result = self.forward_kinematics(current_joint_positions_after)
                if fk_result:
                    current_pos, _ = fk_result
                    pos_error = np.linalg.norm(np.array(current_pos) - np.array(target_position))
                    if pos_error < 1e-3: # Position tolerance
                        print(f"Reached target position (torque control) within tolerance after {step} steps.")
                        # Turn off torques maybe?
                        # self.forward_dynamics([0.0] * self.num_movable_joints)
                        return True
                # else: print("Warning: FK failed during torque control loop.") # Avoid spamming

            print("Failed to reach target position (torque control) within max_steps.")
            # Turn off torques after timeout
            self.forward_dynamics([0.0] * self.num_movable_joints) 
            return False

        else:
            print("Invalid control_mode. Choose 'position' or 'torque'.")
            return False


    def disconnect(self):
        """Disconnects from the physics server."""
        if self.physicsClient >= 0:
             try:
                 p.disconnect()
                 self.physicsClient = -1 # Mark as disconnected
             except Exception as e:
                 print(f"Error during disconnect: {e}")


# --- Example Usage ---
if __name__ == '__main__':
    robot = None # Initialize to None for cleanup
    try:
        # --- Setup ---
        # Make sure 'ur5_description' is installed (`pip install robot_descriptions`)
        robot = RobotArm6DOF(robot_name="ur5_description")

        print(f"\nRobot Initialized: {robot.num_movable_joints} movable joints.")

        robot.reset_robot()
        print("Stepping simulation briefly after reset...")
        for _ in range(50): # Let pybullet settle joints after reset
            p.stepSimulation()
            time.sleep(1./240.)

        # --- Forward Kinematics ---
        print("\n--- Testing Forward Kinematics ---")
        # Ensure the list length matches num_movable_joints (should be 6 for UR5)
        if robot.num_movable_joints == 6:
             joint_angles_fk = [0.5, -1.0, 0.8, 0.2, -0.5, 1.2]
             print(f"Using FK joint angles: {joint_angles_fk}")
             fk_result = robot.forward_kinematics(joint_angles_fk)
             if fk_result:
                 position, orientation = fk_result
                 print(f"FK Result - Position: {np.round(position, 3)}, Orientation: {np.round(orientation, 3)}")
             else:
                 print("FK calculation failed.")
        else:
             print(f"Skipping FK test, expected 6 movable joints, found {robot.num_movable_joints}")


        # --- Inverse Kinematics ---
        print("\n--- Testing Inverse Kinematics ---")
        target_position_ik = [0.4, 0.1, 0.4] # Example target position
        #target_orientation_ik = p.getQuaternionFromEuler([0, np.pi/2, 0]) # Example orientation
        target_orientation_ik = None # Let IK solver choose orientation
        print(f"Target IK Position: {target_position_ik}, Orientation: {target_orientation_ik}")
        joint_angles_ik = robot.inverse_kinematics(target_position_ik, target_orientation_ik, solver=1) # Test null space solver
        if joint_angles_ik:
            print(f"IK Result - Joint Angles: {np.round(joint_angles_ik, 3)}")
        else:
            print("IK calculation failed.")


        # --- Move to Target (Position Control) ---
        print("\n--- Testing Move to Target (Position Control) ---")
        target_pos_1 = [0.3, -0.2, 0.5]
        success = robot.move_to_target(target_pos_1, control_mode='position', max_force=100, position_gain=0.03)
        print(f"Move 1 Success: {success}")
        time.sleep(1)

        # --- Move to Another Target (Position Control)
        print("\n--- Testing Move to Another Target (Position Control) ---")
        target_pos_2 = [0.1, 0.3, 0.3]
        success = robot.move_to_target(target_pos_2, control_mode='position', max_force=100, position_gain=0.03)
        print(f"Move 2 Success: {success}")
        time.sleep(1)

        # --- Inverse Dynamics ---
        # Note: This is mostly for calculation demonstration, not ideal for control loops due to state reset.
        print("\n--- Testing Inverse Dynamics (Calculation Demo) ---")
        if robot.num_movable_joints > 0:
             desired_positions = [0.1] * robot.num_movable_joints
             desired_velocities = [0.0] * robot.num_movable_joints
             desired_accelerations = [0.0] * robot.num_movable_joints # Zero acceleration for quasi-static torque
             required_torques = robot.inverse_dynamics(desired_positions, desired_velocities, desired_accelerations)
             if required_torques:
                 print(f"ID Result - Required Torques (approx): {np.round(required_torques, 3)}")
             else:
                 print("ID calculation failed.")
        else:
            print("Skipping ID test, no movable joints.")

        # --- Move to Target (Torque Control) ---
        print("\n--- Testing Move to Target (Torque Control) ---")
        target_pos_3 = [0.4, 0.0, 0.6]
        # Torque control often needs careful gain tuning!
        success = robot.move_to_target(target_pos_3, control_mode='torque', max_force=50, position_gain=10, velocity_gain=0.5)
        print(f"Move 3 (Torque) Success: {success}")
        time.sleep(2)

        # --- Keep simulation running ---
        print("\nResetting robot. Simulation will continue. Press Ctrl+C to exit.")
        robot.reset_robot()
        # Keep the simulation running for observation
        while True:
            p.stepSimulation()
            time.sleep(1. / 240.)

    except Exception as e:
        print(f"\nAn error occurred: {e}")
        import traceback
        traceback.print_exc()

    finally:
        if robot:
            print("\nDisconnecting from PyBullet.")
            robot.disconnect()