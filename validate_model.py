import torch
import torch.nn as nn # Needed for model definition
import numpy as np
import os
import joblib
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler # Need class definition for loading scaler
from train_surrogate import SurrogateNet

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

# --- Configuration ---
# Directories should match those used in previous phases
DATA_DIR = "synthetic_data"
MODEL_SAVE_DIR = "surrogate_models"

# File paths
MODEL_PATH = os.path.join(MODEL_SAVE_DIR, "surrogate_model.pth") # From Phase 3
X_SCALER_PATH = os.path.join(MODEL_SAVE_DIR, "x_scaler.gz")     # From Phase 3
Y_SCALER_PATH = os.path.join(MODEL_SAVE_DIR, "y_scaler.gz")     # From Phase 3
X_TEST_PATH = os.path.join(DATA_DIR, "X_test.npy")              # From Phase 2
Y_TEST_PATH = os.path.join(DATA_DIR, "y_test_pose.npy")         # From Phase 2

# --- Device Setup ---
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
elif torch.backends.mps.is_available():
     device = torch.device("mps")
     print("Using Apple Metal Performance Shaders (MPS)")
else:
    device = torch.device("cpu")
    print("Using CPU")

# --- Helper Functions ---
def load_model(model_path, device):
    """Loads the trained PyTorch surrogate model."""
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        raise FileNotFoundError(f"Model file not found: {model_path}")
    try:
        # Instantiate the model architecture (sizes must match the saved model)
        model = SurrogateNet(INPUT_SIZE, HIDDEN_1, HIDDEN_2, OUTPUT_SIZE, DROPOUT_RATE).to(device)
        # Load the learned parameters
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval() # Set to evaluation mode
        print(f"Surrogate model loaded successfully from {model_path}")
        return model
    except Exception as e:
        print(f"Error loading model state_dict from {model_path}: {e}")
        raise

def load_scalers(x_scaler_path, y_scaler_path):
    """Loads the fitted X and Y scalers."""
    if not os.path.exists(x_scaler_path) or not os.path.exists(y_scaler_path):
        msg = f"Scaler file(s) not found. Checked:\n - {x_scaler_path}\n - {y_scaler_path}"
        print(f"Error: {msg}")
        raise FileNotFoundError(msg)
    try:
        x_scaler = joblib.load(x_scaler_path)
        y_scaler = joblib.load(y_scaler_path)
        print(f"Scalers loaded successfully.")
        # Basic check to see if scalers seem fitted
        if not hasattr(x_scaler, 'mean_') or not hasattr(y_scaler, 'mean_'):
             print("Warning: Loaded scalers might not be fitted (missing 'mean_' attribute).")
        return x_scaler, y_scaler
    except Exception as e:
        print(f"Error loading scalers: {e}")
        raise

def load_test_data(x_test_path, y_test_path):
    """Loads the test dataset."""
    if not os.path.exists(x_test_path) or not os.path.exists(y_test_path):
        msg = f"Test data file(s) not found. Checked:\n - {x_test_path}\n - {y_test_path}"
        print(f"Error: {msg}")
        raise FileNotFoundError(msg)
    try:
        X_test = np.load(x_test_path).astype(np.float32)
        y_test = np.load(y_test_path).astype(np.float32)
        print(f"Test data loaded successfully: X_test shape {X_test.shape}, y_test shape {y_test.shape}")
        if X_test.shape[0] != y_test.shape[0]:
             print("Warning: Mismatch in number of samples between X_test and y_test.")
        if X_test.shape[1] != 6 or y_test.shape[1] != 6:
             print(f"Warning: Unexpected data dimensions. Expected (N, 6), got X: {X_test.shape}, y: {y_test.shape}")
        return X_test, y_test
    except Exception as e:
        print(f"Error loading test data: {e}")
        raise

# --- Main Execution ---
if __name__ == "__main__":
    print("\n--- Phase 6: Surrogate Model Validation and Testing ---")

    try:
        # 1. Load Model, Scalers, and Data
        model = load_model(MODEL_PATH, device)
        x_scaler, y_scaler = load_scalers(X_SCALER_PATH, Y_SCALER_PATH)
        X_test_orig, y_test_orig = load_test_data(X_TEST_PATH, Y_TEST_PATH)

        # 2. Prepare Data for Inference
        # Scale the input test data using the loaded x_scaler
        X_test_scaled = x_scaler.transform(X_test_orig)
        # Convert scaled input data to PyTorch tensor and move to device
        X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32).to(device)

        # 3. Perform Inference
        print("\nPerforming inference on the test set...")
        predictions_scaled_list = []
        with torch.no_grad():
             # Process in batches if dataset is very large (optional for typical test set sizes)
             # For simplicity, predict all at once if memory allows
             predictions_scaled_tensor = model(X_test_tensor)
             # Move predictions back to CPU and convert to NumPy
             predictions_scaled_np = predictions_scaled_tensor.cpu().numpy()

        print(f"Inference complete. Predictions shape: {predictions_scaled_np.shape}")

        # 4. Unscale Predictions and Targets for Evaluation
        # Inverse transform predictions to get them in original physical units
        predictions_unscaled_np = y_scaler.inverse_transform(predictions_scaled_np)
        # y_test_orig is already in original units, no transformation needed
        targets_unscaled_np = y_test_orig

        # --- 5. Calculate Metrics ---

        # 5.a) Mean Absolute Error (MAE)
        print("\n--- Performance Metrics (Original Scale) ---")
        overall_mae = mean_absolute_error(targets_unscaled_np, predictions_unscaled_np)
        print(f"Overall Mean Absolute Error (MAE): {overall_mae:.6f}")

        # Calculate MAE per dimension (x, y, z, roll, pitch, yaw)
        mae_per_dim = np.mean(np.abs(targets_unscaled_np - predictions_unscaled_np), axis=0)
        print("\nMAE per dimension:")
        print(f"  Position (x, y, z):      [{mae_per_dim[0]:.4f} m, {mae_per_dim[1]:.4f} m, {mae_per_dim[2]:.4f} m]")
        print(f"  Orientation (r, p, y):   [{mae_per_dim[3]:.4f} rad, {mae_per_dim[4]:.4f} rad, {mae_per_dim[5]:.4f} rad]")

        # Calculate average position error in millimeters (as mentioned in MVP doc)
        avg_pos_mae_m = np.mean(mae_per_dim[:3])
        avg_pos_mae_mm = avg_pos_mae_m * 1000
        print(f"\nAverage Position MAE: {avg_pos_mae_m:.4f} m ({avg_pos_mae_mm:.2f} mm)")

        # Compare against target (< 2mm)
        target_mm = 2.0
        if avg_pos_mae_mm < target_mm:
             print(f" -> Meets the MVP target of < {target_mm} mm average position error.")
        else:
             print(f" -> Exceeds the MVP target of < {target_mm} mm average position error.")


        # 5.b) Temporal Consistency Check
        print("\n--- Temporal Consistency Check ---")
        if len(targets_unscaled_np) < 2:
            print("Skipping temporal check: Need at least 2 data points.")
        else:
            sequential_errors = []
            for i in range(1, len(targets_unscaled_np)):
                # Difference between consecutive true poses
                delta_real = targets_unscaled_np[i] - targets_unscaled_np[i-1]
                # Difference between corresponding predicted poses
                delta_pred = predictions_unscaled_np[i] - predictions_unscaled_np[i-1]

                # Error is the norm of the difference between the deltas
                error_norm = np.linalg.norm(delta_pred - delta_real)
                sequential_errors.append(error_norm)

            mean_sequential_error = np.mean(sequential_errors)
            std_sequential_error = np.std(sequential_errors)
            print(f"Mean Sequential Error (Norm of difference in deltas): {mean_sequential_error:.6f}")
            print(f"Std Dev Sequential Error: {std_sequential_error:.6f}")
            # Interpretation: Lower values indicate that the model's predictions change
            # in a way that is consistent with how the real system's pose changes for
            # similar changes in joint angles in the test sequence.

    except FileNotFoundError as e:
        print(f"\nError: A required file was not found. Please ensure previous phases ran successfully.")
        print(f"Details: {e}")
    except Exception as e:
        print(f"\nAn unexpected error occurred during validation: {e}")
        import traceback
        traceback.print_exc()

    print("\n--- Phase 6 Complete ---")