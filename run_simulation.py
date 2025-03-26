# run_simulation.py
# Phase 5: Integrated Simulation System with PyBullet Visualization

import pybullet as p
import pybullet_data
import numpy as np
import torch
import time
import os

# Import classes from previous phases
from robot import RobotArm6DOF # Phase 1
from cognitive_controller import CognitiveController

# --- Configuration ---
ROBOT_NAME = "ur5_description" # Or provide URDF path if needed
# MODEL_PATH = "./surrogate_models/surrogate_model.pth" # From Phase 3 training
MODEL_SAVE_DIR = "surrogate_models"
MODEL_FILENAME = "surrogate_model.pth"
X_SCALER_FILENAME = "x_scaler.gz"
Y_SCALER_FILENAME = "y_scaler.gz"

MODEL_PATH = os.path.join(MODEL_SAVE_DIR, MODEL_FILENAME)
X_SCALER_PATH = os.path.join(MODEL_SAVE_DIR, X_SCALER_FILENAME)
Y_SCALER_PATH = os.path.join(MODEL_SAVE_DIR, Y_SCALER_FILENAME)

# Simulation Parameters
SIMULATION_TIMESTEP = 1. / 240. # PyBullet simulation step frequency
MAX_SIMULATION_STEPS = 1500 # How long to run the simulation (increased slightly)
CONTROL_FREQUENCY = 30 # How often to run the cognitive controller (Hz)
STEPS_PER_CONTROL = int(1.0 / (SIMULATION_TIMESTEP * CONTROL_FREQUENCY))

# Target Pose [x, y, z, roll, pitch, yaw] (example)
TARGET_POSE_6D = np.array([0.4, 0.1, 0.5, 0.0, np.pi/2, 0.0], dtype=np.float32)
TARGET_POS_3D = TARGET_POSE_6D[:3] # Extract just [x, y, z] for visualization

# PyBullet Position Control Gains (Tune these)
POSITION_GAIN_KP = 0.03
VELOCITY_GAIN_KD = 1.0
MAX_JOINT_FORCE = 60

# Visualization Parameters (using PyBullet Debug Items)
TARGET_MARKER_COLOR = [1, 0, 0] # Red
TARGET_MARKER_SIZE = 0.05 # Size of the cross marker lines
PATH_COLOR = [0, 0, 1] # Blue
PATH_LINE_WIDTH = 2
VIS_UPDATE_FREQUENCY = 20 # Hz (Update visualization more frequently)
STEPS_PER_VIS_UPDATE = int(1.0 / (SIMULATION_TIMESTEP * VIS_UPDATE_FREQUENCY))


# --- Device Setup (same as Phase 4) ---
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
     device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"Using device: {device}")


# --- Helper function to draw a cross marker ---
def draw_target_marker(position, size, color, physicsClientId):
    """Draws a 3D cross marker using PyBullet debug lines."""
    p.addUserDebugLine(position - [size/2, 0, 0], position + [size/2, 0, 0], color, lineWidth=2, lifeTime=0, physicsClientId=physicsClientId)
    p.addUserDebugLine(position - [0, size/2, 0], position + [0, size/2, 0], color, lineWidth=2, lifeTime=0, physicsClientId=physicsClientId)
    p.addUserDebugLine(position - [0, 0, size/2], position + [0, 0, size/2], color, lineWidth=2, lifeTime=0, physicsClientId=physicsClientId)

# --- Main Simulation ---
if __name__ == "__main__":
    print("--- Phase 5: Integrated Simulation (PyBullet Visualization) ---")
    physicsClient = -1 # Initialize client ID for cleanup
    robot_sim = None
    actual_pos_history = [] # Store the path points
    debug_line_ids = [] # Store IDs of drawn path segments

    try:
        # 1. Initialize Simulation Environment (GUI mode)
        print("Connecting to PyBullet GUI...")
        physicsClient = p.connect(p.GUI)
        if physicsClient < 0:
             raise RuntimeError("Failed to connect to PyBullet GUI.")
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 1)
        p.resetDebugVisualizerCamera(cameraDistance=1.5, cameraYaw=30, cameraPitch=-20, cameraTargetPosition=[0,0,0.3])
        p.setGravity(0, 0, -9.81) # Set gravity explicitly
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        planeId = p.loadURDF("plane.urdf") # Load ground plane


        # 2. Initialize Robot Arm (pass the GUI client)
        print(f"Loading robot: {ROBOT_NAME}")
        # Ensure RobotArm6DOF uses the *passed* client ID if modified.
        robot_sim = RobotArm6DOF(robot_name=ROBOT_NAME, physicsClientId=physicsClient)

        # 3. Initialize Cognitive Controller
        print("Initializing Cognitive Controller...")
        controller = CognitiveController(MODEL_PATH, X_SCALER_PATH, Y_SCALER_PATH, device)

        # 4. Initialize Visualization (Draw static target marker)
        print("Drawing target marker in PyBullet GUI...")
        draw_target_marker(TARGET_POS_3D, TARGET_MARKER_SIZE, TARGET_MARKER_COLOR, physicsClient)

        # 5. Reset Robot to Initial State
        initial_joint_angles = [0.1] * robot_sim.num_movable_joints
        print(f"Resetting robot to initial angles: {np.round(initial_joint_angles, 2)}")
        robot_sim.reset_robot(initial_joint_angles)
        # Let simulation settle
        for _ in range(100):
            p.stepSimulation()
            time.sleep(SIMULATION_TIMESTEP)
        
        # Get initial position to start the path history
        initial_fk_result = robot_sim.forward_kinematics(initial_joint_angles)
        if initial_fk_result:
            actual_pos_history.append(np.array(initial_fk_result[0]))


        # --- 6. Simulation Loop ---
        print("\n--- Starting Simulation Loop ---")
        print(f"Running for {MAX_SIMULATION_STEPS} steps.")
        print(f"Control decision every {STEPS_PER_CONTROL} steps ({CONTROL_FREQUENCY} Hz).")
        print(f"Visualization update every {STEPS_PER_VIS_UPDATE} steps ({VIS_UPDATE_FREQUENCY} Hz).")

        for step in range(MAX_SIMULATION_STEPS):
            try:
                # --- Control Decision Step ---
                if step % STEPS_PER_CONTROL == 0:
                    current_joint_angles, _, _ = robot_sim.get_joint_states()
                    current_joint_angles_np = np.array(current_joint_angles, dtype=np.float32)

                    correction_signal = controller.make_decision(current_joint_angles_np, TARGET_POSE_6D)

                    if correction_signal is not None:
                        target_joint_angles = current_joint_angles_np + correction_signal
                        target_joint_angles = np.clip(target_joint_angles,
                                                      robot_sim.joint_lower_limits,
                                                      robot_sim.joint_upper_limits)

                        p.setJointMotorControlArray(
                            bodyUniqueId=robot_sim.robotId,
                            jointIndices=robot_sim.movable_joint_indices,
                            controlMode=p.POSITION_CONTROL,
                            targetPositions=target_joint_angles,
                            forces=[MAX_JOINT_FORCE] * robot_sim.num_movable_joints,
                            positionGains=[POSITION_GAIN_KP] * robot_sim.num_movable_joints,
                            velocityGains=[VELOCITY_GAIN_KD] * robot_sim.num_movable_joints
                        )
                    # else: Handle controller failure if needed

                # --- Step Simulation ---
                p.stepSimulation()

                # --- Visualization Update Step ---
                if step % STEPS_PER_VIS_UPDATE == 0 and len(actual_pos_history) > 0:
                     # Get current actual joint states AFTER stepping
                     current_joint_positions, _, _ = robot_sim.get_joint_states()
                     # Calculate actual end-effector position using FK
                     fk_result = robot_sim.forward_kinematics(current_joint_positions)
                     
                     if fk_result:
                         current_pos = np.array(fk_result[0])
                         # Get the previous position from history
                         prev_pos = actual_pos_history[-1]
                         
                         # Draw a line segment from the previous position to the current one
                         line_id = p.addUserDebugLine(prev_pos, current_pos, PATH_COLOR, lineWidth=PATH_LINE_WIDTH, lifeTime=0, physicsClientId=physicsClient)
                         debug_line_ids.append(line_id) # Store ID if you want to remove lines later
                         
                         # Add current position to history for the next segment
                         actual_pos_history.append(current_pos)

                         # Optional: Limit history size and remove old lines to prevent slowdown
                         # MAX_PATH_LINES = 500
                         # if len(debug_line_ids) > MAX_PATH_LINES:
                         #     line_to_remove = debug_line_ids.pop(0)
                         #     p.removeUserDebugItem(line_to_remove, physicsClientId=physicsClient)
                         #     actual_pos_history.pop(0) # Keep history and line IDs in sync


                # --- Timing ---
                time.sleep(SIMULATION_TIMESTEP)
                
                # Optional: Target reached check
                if len(actual_pos_history) > 0:
                     pos_error = np.linalg.norm(actual_pos_history[-1] - TARGET_POS_3D)
                     if pos_error < 0.01: # Tolerance in meters
                         print(f"\nTarget reached within tolerance after {step} steps!")
                         break


            except KeyboardInterrupt:
                print("\nSimulation interrupted by user.")
                break

        print("\n--- Simulation Loop Finished ---")
        print("Simulation window will remain open. Close it manually or press Ctrl+C again in terminal.")
        
        # Keep the simulation running until manually closed or interrupted again
        while True:
            try:
                # Keep rendering
                p.stepSimulation() # Or just sleep if no physics needed
                time.sleep(0.01)
            except KeyboardInterrupt:
                break
            except p.error as e: # Catch pybullet errors if window is closed
                print(f"PyBullet error (likely window closed): {e}")
                break


    except FileNotFoundError as e:
         print(f"\nError: {e}")
         print("Ensure model file exists and previous phases were run.")
    except RuntimeError as e:
         print(f"\nRuntime Error: {e}")
    except Exception as e:
        print(f"\nAn unexpected error occurred during simulation: {e}")
        import traceback
        traceback.print_exc()

    finally:
        # --- Cleanup ---
        # Check if physicsClient was successfully created before disconnecting
        if 'physicsClient' in locals() and isinstance(physicsClient, int) and physicsClient >= 0:
            try:
                # Check if still connected before disconnecting
                 if p.isConnected(physicsClientId=physicsClient):
                    print("Disconnecting from PyBullet.")
                    p.disconnect(physicsClientId=physicsClient)
                 else:
                     print("PyBullet already disconnected.")

            except TypeError: # Handle case where physicsClient might not be an int
                 print("Invalid physicsClient ID, cannot disconnect.")
            except p.error as e:
                 print(f"PyBullet error during disconnect (maybe already disconnected?): {e}")
            except Exception as e:
                 print(f"Error during PyBullet disconnect: {e}")
        else:
            print("No valid PyBullet connection to disconnect.")


    print("\n--- Phase 5 Complete ---")