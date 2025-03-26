import pybullet as p
import numpy as np
from scipy.spatial.transform import Rotation # For handling orientations
import os
import time
from robot import RobotArm6DOF # Assuming your Phase 1 code is in robot_arm.py

# --- Configuration ---
ROBOT_NAME = "ur5_description" # Or provide URDF path if needed
# URDF_PATH = "path/to/your/6dof_robot.urdf" # Alternative

NUM_TRAIN_SAMPLES = 10000
NUM_TEST_SAMPLES = 2000

# Noise Parameters
POS_NOISE_STD_DEV = 0.001 # Standard deviation for position noise (meters)
ORN_NOISE_STD_DEV = 0.01  # Standard deviation for orientation noise (radians)
EULER_CONVENTION = 'xyz' # Convention for Euler angles ('xyz', 'zyx', etc.)

OUTPUT_DIR = "synthetic_data"
SEED = 42 # for reproducibility

# --- Helper Function ---

def generate_dataset(robot, num_samples, pos_noise_std, orn_noise_std, euler_convention='xyz'):
    """
    Generates a dataset of (joint_angles, noisy_end_effector_pose).

    Args:
        robot (RobotArm6DOF): Instance of the robot arm simulator.
        num_samples (int): Number of data points to generate.
        pos_noise_std (float): Standard deviation for additive Gaussian noise on position.
        orn_noise_std (float): Standard deviation for additive Gaussian noise on Euler angles.
        euler_convention (str): Euler angle convention (e.g., 'xyz', 'zyx').

    Returns:
        tuple: (X_data, Y_data)
               X_data (np.ndarray): Array of joint angles (shape: num_samples x num_movable_joints).
               Y_data (np.ndarray): Array of noisy end-effector poses
                                    (shape: num_samples x 6 -> [x, y, z, roll, pitch, yaw]).
    """
    X_data = []
    Y_data = []
    
    # Get joint limits for sampling
    lower_limits = robot.joint_lower_limits
    upper_limits = robot.joint_upper_limits
    num_joints = robot.num_movable_joints

    print(f"Generating {num_samples} samples...")
    start_time = time.time()

    generated_count = 0
    while generated_count < num_samples:
        # 1. Sample Random Joint Angles within limits
        joint_angles = np.random.uniform(low=lower_limits, high=upper_limits, size=num_joints)

        # 2. Calculate Forward Kinematics
        fk_result = robot.forward_kinematics(joint_angles.tolist())

        if fk_result is None:
            print("Warning: FK failed for a sample, skipping.")
            continue # Skip this sample if FK failed

        pos_fk, orn_quat_fk = fk_result

        # 3. Add Synthetic Sensor Noise
        # Position Noise
        pos_noise = np.random.normal(0, pos_noise_std, 3)
        noisy_pos = np.array(pos_fk) + pos_noise

        # Orientation Noise
        try:
            # Convert quaternion to Euler angles
            rotation = Rotation.from_quat(orn_quat_fk)
            euler_angles = rotation.as_euler(euler_convention)

            # Add noise to Euler angles
            orn_noise = np.random.normal(0, orn_noise_std, 3)
            noisy_euler = euler_angles + orn_noise

            # Optional: Handle angle wrapping if necessary, though often not critical for training data range
            # noisy_euler = np.unwrap(noisy_euler) # Might be needed depending on convention/range

        except Exception as e:
            print(f"Warning: Orientation conversion/noise addition failed: {e}. Skipping sample.")
            continue # Skip sample if rotation math fails

        # 4. Combine noisy position and orientation (as Euler) into 6D pose vector
        noisy_pose_6d = np.concatenate([noisy_pos, noisy_euler])

        # 5. Store data
        X_data.append(joint_angles)
        Y_data.append(noisy_pose_6d)
        generated_count += 1

        if generated_count % (num_samples // 10) == 0:
             print(f"  Generated {generated_count}/{num_samples} samples...")

    end_time = time.time()
    print(f"Finished generating {num_samples} samples in {end_time - start_time:.2f} seconds.")

    return np.array(X_data), np.array(Y_data)

# --- Main Execution ---
if __name__ == "__main__":
    print("--- Phase 2: Synthetic Data Generation ---")

    # Set random seed for reproducibility
    np.random.seed(SEED)

    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Output directory: {OUTPUT_DIR}")

    robot_sim = None # Initialize for finally block
    try:
        # Initialize Robot Simulation in DIRECT mode (no GUI)
        print(f"Initializing simulation for robot: {ROBOT_NAME}")
        # --- Critical: Connect in DIRECT mode for speed ---
        physicsClient = p.connect(p.DIRECT)
        if physicsClient < 0:
             raise RuntimeError("Failed to connect to PyBullet in DIRECT mode.")

        # Pass the existing physics client ID to the RobotArm6DOF constructor
        # Modify RobotArm6DOF slightly to accept an optional physicsClientId
        # (See modification note below)
        # If RobotArm6DOF always creates its own connection, we need to adjust.
        # For now, assuming RobotArm6DOF can use an existing client or we manage it here.

        # --- Simplified Approach: Let RobotArm6DOF manage its connection (as originally written) ---
        # Need to modify RobotArm6DOF slightly OR manage client externally.
        # Let's assume the original RobotArm6DOF always calls p.connect().
        # We will create a *temporary* instance just for data generation.

        # Connect PyBullet in DIRECT mode *before* creating the RobotArm6DOF instance
        # if RobotArm6DOF checks p.isConnected() or similar.
        # Best practice: Modify RobotArm6DOF to accept a client ID.
        # Quick fix: Instantiate RobotArm6DOF but ensure it uses DIRECT mode.

        # --- Assuming RobotArm6DOF constructor is modified like this: ---
        # class RobotArm6DOF:
        #     def __init__(self, ..., physicsClientId=None):
        #         if physicsClientId is None:
        #             self.physicsClient = p.connect(p.GUI) # Default to GUI if not provided
        #         else:
        #             self.physicsClient = physicsClientId # Use provided client
        #         # ... rest of init
        # robot_sim = RobotArm6DOF(robot_name=ROBOT_NAME, physicsClientId=physicsClient) # Preferred

        # --- OR If RobotArm6DOF cannot accept client ID, modify its connect line: ---
        # Change `p.connect(p.GUI)` to `p.connect(p.DIRECT)` inside RobotArm6DOF
        # for this script's purpose. Or, manage the instance carefully.

        # --- Safest approach if RobotArm6DOF isn't modified: ---
        print("Creating RobotArm6DOF instance (ensure it uses p.DIRECT internally for this script)...")
        # Make sure the RobotArm6DOF class code used by this script has p.connect(p.DIRECT)
        robot_sim = RobotArm6DOF(robot_name=ROBOT_NAME) # Use robot_name or urdf_path
        # Verify it connected in the right mode (optional check)
        # connection_info = p.getConnectionInfo(robot_sim.physicsClient)
        # if connection_info['connectionMethod'] != p.DIRECT:
        #    print("Warning: RobotArm6DOF did not connect in DIRECT mode!")


        # --- Generate Training Data ---
        print("\nGenerating Training Data...")
        X_train, y_train = generate_dataset(robot_sim, NUM_TRAIN_SAMPLES, POS_NOISE_STD_DEV, ORN_NOISE_STD_DEV, EULER_CONVENTION)
        print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")

        # --- Generate Testing Data ---
        print("\nGenerating Testing Data...")
        X_test, y_test = generate_dataset(robot_sim, NUM_TEST_SAMPLES, POS_NOISE_STD_DEV, ORN_NOISE_STD_DEV, EULER_CONVENTION)
        print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")

        # --- Save Data ---
        print("\nSaving datasets...")
        np.save(os.path.join(OUTPUT_DIR, "X_train.npy"), X_train)
        np.save(os.path.join(OUTPUT_DIR, "y_train_pose.npy"), y_train)
        np.save(os.path.join(OUTPUT_DIR, "X_test.npy"), X_test)
        np.save(os.path.join(OUTPUT_DIR, "y_test_pose.npy"), y_test)
        print("Datasets saved successfully.")

    except Exception as e:
        print(f"\nAn error occurred during data generation: {e}")
        import traceback
        traceback.print_exc()

    finally:
        # --- Disconnect ---
        if robot_sim:
             # If RobotArm6DOF manages its own client ID:
             if hasattr(robot_sim, 'physicsClient') and robot_sim.physicsClient >= 0:
                 try:
                     print("\nDisconnecting from PyBullet.")
                     p.disconnect(physicsClientId=robot_sim.physicsClient)
                 except Exception as e:
                      print(f"Error during disconnect: {e}")
             # OR if we managed the client ID externally:
             # elif physicsClient >= 0:
             #     p.disconnect(physicsClientId=physicsClient)
        elif 'physicsClient' in locals() and physicsClient >= 0:
             # Fallback if robot_sim failed but client was created
             try:
                  print("\nDisconnecting from PyBullet (fallback).")
                  p.disconnect(physicsClientId=physicsClient)
             except Exception as e:
                 print(f"Error during fallback disconnect: {e}")

    print("\n--- Data Generation Complete ---")