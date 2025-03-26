# cognitive_controller.py (Improved with Scaling)

import torch
import torch.nn as nn
import numpy as np
import os
from collections import deque
import joblib # Required for loading scikit-learn scalers
from sklearn.preprocessing import StandardScaler # Required by joblib to unpickle scaler objects
from train_surrogate import SurrogateNet

# --- Configuration ---
# Ensure these paths point to the files saved by train_surrogate.py
MODEL_SAVE_DIR = "surrogate_models"
MODEL_FILENAME = "surrogate_model.pth"
X_SCALER_FILENAME = "x_scaler.gz"
Y_SCALER_FILENAME = "y_scaler.gz"

MODEL_PATH = os.path.join(MODEL_SAVE_DIR, MODEL_FILENAME)
X_SCALER_PATH = os.path.join(MODEL_SAVE_DIR, X_SCALER_FILENAME)
Y_SCALER_PATH = os.path.join(MODEL_SAVE_DIR, Y_SCALER_FILENAME)

INPUT_SIZE = 6
OUTPUT_SIZE = 6
HIDDEN_1 = 128
HIDDEN_2 = 256
DROPOUT_RATE = 0.2
LEARNING_RATE = 0.001
BATCH_SIZE = 32
EPOCHS = 100
VALIDATION_SPLIT = 0.2
EARLY_STOPPING_PATIENCE = 10 # Re-enable this
SEED = 42

# --- Controller Parameters ---
MEMORY_SIZE = 10 # How many past errors to remember for the 'I' term
KP = 0.8 # Proportional gain
KI = 0.2 # Integral gain
# Note: These gains will likely need significant tuning now that the error signal is correctly scaled.

# --- Device Setup ---
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"Cognitive Controller: Using GPU: {torch.cuda.get_device_name(0)}")
elif torch.backends.mps.is_available():
     device = torch.device("mps")
     print("Cognitive Controller: Using Apple Metal Performance Shaders (MPS)")
else:
    device = torch.device("cpu")
    print("Cognitive Controller: Using CPU")

# --- Cognitive Controller Class (Improved) ---
class CognitiveController:
    def __init__(self, surrogate_model_path, x_scaler_path, y_scaler_path, device):
        """
        Initializes the Cognitive Controller.

        Args:
            surrogate_model_path (str): Path to the trained surrogate model (.pth file).
            x_scaler_path (str): Path to the saved input scaler (.gz file).
            y_scaler_path (str): Path to the saved output scaler (.gz file).
            device (torch.device): The device (CPU/GPU/MPS) to run inference on.
        """
        self.device = device
        self.surrogate_model = self._load_model(surrogate_model_path)
        self.x_scaler, self.y_scaler = self._load_scalers(x_scaler_path, y_scaler_path)
        self.error_memory = deque(maxlen=MEMORY_SIZE)
        print("Cognitive Controller initialized with model and scalers.")

    def _load_model(self, model_path):
        """Loads the trained PyTorch surrogate model."""
        if not os.path.exists(model_path):
            print(f"Error: Model file not found at {model_path}")
            raise FileNotFoundError(f"Model file not found: {model_path}")
        try:
            # Instantiate the model architecture (must match saved model's arch)
            model = SurrogateNet(INPUT_SIZE, HIDDEN_1, HIDDEN_2, OUTPUT_SIZE, DROPOUT_RATE).to(device) # Using default sizes from definition
            model.load_state_dict(torch.load(model_path, map_location=self.device))
            model.eval()
            print(f"Surrogate model loaded successfully from {model_path}")
            return model
        except Exception as e:
            print(f"Error loading model state_dict from {model_path}: {e}")
            raise

    def _load_scalers(self, x_scaler_path, y_scaler_path):
        """Loads the fitted X and Y scalers saved during training."""
        if not os.path.exists(x_scaler_path):
            raise FileNotFoundError(f"Input scaler not found: {x_scaler_path}")
        if not os.path.exists(y_scaler_path):
            raise FileNotFoundError(f"Output scaler not found: {y_scaler_path}")
        try:
            x_scaler = joblib.load(x_scaler_path)
            y_scaler = joblib.load(y_scaler_path)
            # Basic check: Ensure scalers appear to be fitted
            if not hasattr(x_scaler, 'mean_') or not hasattr(y_scaler, 'mean_'):
                print("Warning: Loaded scalers might not be fitted (missing 'mean_' attribute).")
            print("Input and output scalers loaded successfully.")
            return x_scaler, y_scaler
        except Exception as e:
            print(f"Error loading scalers: {e}")
            raise

    def make_decision(self, current_joint_state, target_pose):
        """
        Makes a decision (calculates a correction) based on the current state and target pose,
        using the scaled surrogate model.

        Args:
            current_joint_state (np.ndarray): Current joint angles (shape: (6,)).
            target_pose (np.ndarray): Desired target end-effector pose in PHYSICAL UNITS (shape: (6,) -> [x, y, z, r, p, y]).

        Returns:
            np.ndarray: Calculated correction signal based on PI control of the pose error (shape: (6,)).
                        Returns None if prediction fails.
        """
        if not isinstance(current_joint_state, np.ndarray):
            current_joint_state = np.array(current_joint_state, dtype=np.float32)
        if not isinstance(target_pose, np.ndarray):
            target_pose = np.array(target_pose, dtype=np.float32)

        if current_joint_state.shape != (6,) or target_pose.shape != (6,):
             print(f"Error: Invalid input shapes. current_joint_state: {current_joint_state.shape}, target_pose: {target_pose.shape}")
             return None

        # 1. Scale Input: Prepare current joint state for the model
        try:
            # Reshape for scaler (expects 2D array: [n_samples, n_features])
            current_joint_state_reshaped = current_joint_state.reshape(1, -1)
            # Scale using the loaded x_scaler
            input_scaled = self.x_scaler.transform(current_joint_state_reshaped)
            # Convert to PyTorch tensor and move to the correct device
            input_tensor = torch.tensor(input_scaled, dtype=torch.float32).to(self.device)
        except Exception as e:
            print(f"Error during input scaling: {e}")
            return None

        # 2. Predict End-Effector Pose (Scaled) using Surrogate Model
        try:
            with torch.no_grad(): # Disable gradient calculation for inference
                predicted_pose_scaled_tensor = self.surrogate_model(input_tensor)
            # Move prediction back to CPU and convert to NumPy array
            predicted_pose_scaled = predicted_pose_scaled_tensor.cpu().numpy() # Shape is likely (1, 6)
        except Exception as e:
            print(f"Error during surrogate model prediction: {e}")
            return None

        # 3. Unscale Output: Convert prediction back to physical units
        try:
            # Unscale using the loaded y_scaler (expects 2D, provides 2D output)
            predicted_pose_unscaled_reshaped = self.y_scaler.inverse_transform(predicted_pose_scaled)
            # Flatten back to 1D array for error calculation
            predicted_pose_unscaled = predicted_pose_unscaled_reshaped.flatten() # Shape (6,)
        except Exception as e:
            print(f"Error during output unscaling: {e}")
            return None

        # 4. Calculate Error (Now in Physical Units)
        # Error = Target Pose - Predicted Pose (both in meters/radians)
        error = target_pose - predicted_pose_unscaled

        # 5. Update Error Memory and Calculate Integral Term
        self.error_memory.append(error)
        if len(self.error_memory) > 0:
            mean_error = np.mean(np.array(self.error_memory), axis=0)
        else:
            mean_error = np.zeros_like(error)

        # 6. Calculate Correction (Simple PI-like Control)
        # Correction is based on the error in TASK SPACE (pose).
        # How this correction is *applied* (e.g., added to joints, used with Jacobian/IK)
        # depends on the integration phase (run_simulation.py). This function just calculates it.
        correction = (KP * error) + (KI * mean_error)

        return correction

# --- Example Usage (Updated) ---
if __name__ == "__main__":
    print("--- Phase 4: Cognitive Controller Demonstration (Improved with Scaling) ---")

    # Check if required files exist before proceeding
    required_files = [MODEL_PATH, X_SCALER_PATH, Y_SCALER_PATH]
    files_missing = False
    for f_path in required_files:
        if not os.path.exists(f_path):
            print(f"Error: Required file not found: {f_path}")
            files_missing = True
    if files_missing:
        print("Please ensure Phase 2 (generate_data.py) and Phase 3 (train_surrogate.py) were run successfully.")
        exit(1) # Stop execution if files are missing

    try:
        # 1. Initialize Controller (loads model AND scalers)
        controller = CognitiveController(MODEL_PATH, X_SCALER_PATH, Y_SCALER_PATH, device)

        # 2. Define Example Target and Initial State
        target_ee_pose = np.array([0.5, 0.2, 0.4, 0.1, -0.2, 0.3], dtype=np.float32)
        current_robot_joints = np.zeros(6, dtype=np.float32)

        print(f"\nTarget End-Effector Pose: {np.round(target_ee_pose, 3)}")
        print(f"Initial Joint Angles: {np.round(current_robot_joints, 3)}")

        # 3. Simple Simulation Loop (Mimicking MVP Example)
        num_steps = 5 # Reduced steps for clearer demo output
        print(f"\nRunning simple simulation loop for {num_steps} steps...")
        print("-" * 60)

        for step in range(num_steps):
            print(f"Step {step+1}:")
            print(f"  Current Joints (Physical): {np.round(current_robot_joints, 3)}")

            # --- Get Scaled Input for verification ---
            input_scaled_display = controller.x_scaler.transform(current_robot_joints.reshape(1,-1))
            print(f"  Current Joints (Scaled for Model): {np.round(input_scaled_display, 3)}")

            # Make decision (predict, calculate error, get correction)
            correction_signal = controller.make_decision(current_robot_joints, target_ee_pose)

            if correction_signal is None:
                 print("  Decision making failed. Stopping simulation.")
                 break

            # --- Get Unscaled Prediction for verification ---
            # (This involves re-running prediction, just for display)
            with torch.no_grad():
                input_tensor_display = torch.tensor(input_scaled_display, dtype=torch.float32).to(device)
                pred_scaled_display = controller.surrogate_model(input_tensor_display).cpu().numpy()
                pred_unscaled_display = controller.y_scaler.inverse_transform(pred_scaled_display).flatten()
            print(f"  Predicted Pose (Unscaled): {np.round(pred_unscaled_display, 3)}")
            print(f"  Target Pose (Physical):    {np.round(target_ee_pose, 3)}")

            error_display = target_ee_pose - pred_unscaled_display
            print(f"  Calculated Error:          {np.round(error_display, 4)}")
            print(f"  Calculated Correction:     {np.round(correction_signal, 4)}")

            # Apply correction directly to joint angles (as in original MVP's implied usage)
            current_robot_joints += correction_signal

            # Optional: Add clipping based on realistic joint limits if available
            # current_robot_joints = np.clip(current_robot_joints, lower_limits, upper_limits)

            print(f"  Updated Joints (Physical): {np.round(current_robot_joints, 3)}")
            print("-" * 60)

        print("\n--- Cognitive Controller Demonstration Complete ---")

    except FileNotFoundError:
        # Error already printed if files are missing
        pass
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()