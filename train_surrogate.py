# train_surrogate.py (Revised for Scaling and Correct Saving Paths)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import os
import time
import matplotlib.pyplot as plt # For plotting loss
from sklearn.preprocessing import StandardScaler # Import StandardScaler
import joblib # To save the scalers

# --- Configuration ---
DATA_DIR = "synthetic_data"
MODEL_SAVE_DIR = "surrogate_models" # The directory to save outputs

# --- Correctly construct paths within MODEL_SAVE_DIR ---
MODEL_FILENAME = "surrogate_model.pth"
X_SCALER_FILENAME = "x_scaler.gz"
Y_SCALER_FILENAME = "y_scaler.gz"

MODEL_SAVE_PATH = os.path.join(MODEL_SAVE_DIR, MODEL_FILENAME)
X_SCALER_PATH = os.path.join(MODEL_SAVE_DIR, X_SCALER_FILENAME)
Y_SCALER_PATH = os.path.join(MODEL_SAVE_DIR, Y_SCALER_FILENAME)

# --- Create the output directory if it doesn't exist ---
# Corrected: Create the directory specified by MODEL_SAVE_DIR
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
print(f"Model and scalers will be saved in: {os.path.abspath(MODEL_SAVE_DIR)}")


# ... (Model Architecture, Training Params, Device Setup, Seed setting remain the same) ...
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

# --- Set Device ---
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
elif torch.backends.mps.is_available():
     device = torch.device("mps")
     print("Using Apple Metal Performance Shaders (MPS)")
else:
    device = torch.device("cpu")
    print("Using CPU")

# Set random seed for reproducibility
torch.manual_seed(SEED)
np.random.seed(SEED)
if device.type == 'cuda':
    torch.cuda.manual_seed_all(SEED)


# --- Dataset Class (Now expects scaled data) ---
class RobotPoseDataset(Dataset):
    """Custom Dataset for loading SCALED robot joint angles and poses."""
    # Modified to directly accept numpy arrays (scaling happens before dataset creation)
    def __init__(self, X_scaled, y_scaled, device=torch.device("cpu")):
        self.X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(device)
        self.y_tensor = torch.tensor(y_scaled, dtype=torch.float32).to(device)

        if self.X_tensor.shape[0] != self.y_tensor.shape[0]:
             raise ValueError("Mismatch in number of samples between X and y data.")
        # Shapes should still match original input/output sizes logically
        if self.X_tensor.shape[1] != INPUT_SIZE:
             raise ValueError(f"Input data feature size mismatch. Expected {INPUT_SIZE}, got {self.X_tensor.shape[1]}")
        if self.y_tensor.shape[1] != OUTPUT_SIZE:
             raise ValueError(f"Output data feature size mismatch. Expected {OUTPUT_SIZE}, got {self.y_tensor.shape[1]}")

    def __len__(self):
        return len(self.X_tensor)

    def __getitem__(self, idx):
        return self.X_tensor[idx], self.y_tensor[idx]

# --- Surrogate Model Definition (No change needed here) ---
class SurrogateNet(nn.Module):
    def __init__(self, input_size, hidden1_size, hidden2_size, output_size, dropout_rate):
        super(SurrogateNet, self).__init__()
        self.layer_1 = nn.Linear(input_size, hidden1_size)
        self.relu_1 = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.layer_2 = nn.Linear(hidden1_size, hidden2_size)
        self.relu_2 = nn.ReLU()
        self.layer_3 = nn.Linear(hidden2_size, output_size)

    def forward(self, x):
        x = self.layer_1(x)
        x = self.relu_1(x)
        x = self.dropout(x)
        x = self.layer_2(x)
        x = self.relu_2(x)
        x = self.layer_3(x)
        return x

# --- Training Function (No change needed here, operates on scaled data) ---
# Uses the globally defined MODEL_SAVE_PATH
def train_model(model, train_loader, val_loader, criterion, optimizer, epochs, device, patience):
    """Trains the model with validation and early stopping."""
    best_val_loss = float('inf')
    epochs_no_improve = 0
    train_losses = []
    val_losses = []

    print("\n--- Starting Training ---")
    start_time_total = time.time()

    for epoch in range(epochs):
        model.train()
        running_train_loss = 0.0
        epoch_start_time = time.time()

        for i, (inputs, targets) in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_train_loss += loss.item() * inputs.size(0)

        # Adjust for SubsetRandomSampler len (correct calculation)
        epoch_train_loss = running_train_loss / len(train_loader.sampler) if train_loader.sampler else running_train_loss / len(train_loader.dataset)


        model.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                running_val_loss += loss.item() * inputs.size(0)

        # Adjust for SubsetRandomSampler len (correct calculation)
        epoch_val_loss = running_val_loss / len(val_loader.sampler) if val_loader.sampler else running_val_loss / len(val_loader.dataset)

        val_losses.append(epoch_val_loss)
        train_losses.append(epoch_train_loss) # Append train loss here for consistency
        epoch_duration = time.time() - epoch_start_time

        print(f"Epoch [{epoch+1}/{epochs}] - Duration: {epoch_duration:.2f}s - Train Loss: {epoch_train_loss:.6f} - Val Loss: {epoch_val_loss:.6f}")

        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            epochs_no_improve = 0
            # Save using the corrected MODEL_SAVE_PATH
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"  Validation loss improved. Saving model to {MODEL_SAVE_PATH}")
        else:
            epochs_no_improve += 1
            print(f"  Validation loss did not improve for {epochs_no_improve} epoch(s).")

        # --- Re-enable Early Stopping ---
        if epochs_no_improve >= patience:
            print(f"\nEarly stopping triggered after {epoch+1} epochs.")
            break # Stop training

    end_time_total = time.time()
    print(f"\n--- Training Finished --- Total Time: {(end_time_total - start_time_total)/60:.2f} minutes")
    # Return only the losses up to the point of stopping
    # Corrected return slicing
    effective_epochs = epoch + 1
    return train_losses[:effective_epochs], val_losses[:effective_epochs]


# --- Evaluation Function (MODIFIED to use inverse transform) ---
# Uses the globally defined MODEL_SAVE_PATH for loading
def evaluate_model(model, test_loader, device, y_scaler): # Pass y_scaler
    """Evaluates the model on the test set and calculates MAE on original scale."""
    model.eval()
    all_predictions_scaled = []
    all_targets_scaled = []

    with torch.no_grad():
        for inputs, targets_scaled in test_loader:
            outputs_scaled = model(inputs)
            all_predictions_scaled.append(outputs_scaled.cpu().numpy())
            all_targets_scaled.append(targets_scaled.cpu().numpy())

    predictions_scaled_np = np.concatenate(all_predictions_scaled, axis=0)
    targets_scaled_np = np.concatenate(all_targets_scaled, axis=0)

    # --- Inverse Transform to Original Scale ---
    try:
        # Ensure scalers were fitted before trying inverse_transform
        if not hasattr(y_scaler, 'mean_') or not hasattr(y_scaler, 'scale_'):
             raise ValueError("Y scaler has not been fitted yet.")

        predictions_unscaled_np = y_scaler.inverse_transform(predictions_scaled_np)
        targets_unscaled_np = y_scaler.inverse_transform(targets_scaled_np)
    except Exception as e:
        print(f"Error during inverse scaling for evaluation: {e}")
        # Fallback to reporting MAE on scaled data if unscaling fails
        print("Reporting MAE on SCALED data due to error.")
        predictions_unscaled_np = predictions_scaled_np
        targets_unscaled_np = targets_scaled_np
        scale_info = "(SCALED)"
    else:
        scale_info = "(Original Scale)"


    # Calculate MAE
    mae = np.mean(np.abs(predictions_unscaled_np - targets_unscaled_np))
    mae_per_dim = np.mean(np.abs(predictions_unscaled_np - targets_unscaled_np), axis=0)

    print(f"\n--- Evaluation on Test Set {scale_info} ---")
    print(f"Overall Mean Absolute Error (MAE): {mae:.6f}")
    print(f"MAE per dimension (x, y, z, r, p, y):")
    # Adjust printing based on whether data is scaled or not
    if scale_info == "(Original Scale)":
        print(f"  Position (xyz): {mae_per_dim[0]:.4f}m, {mae_per_dim[1]:.4f}m, {mae_per_dim[2]:.4f}m")
        print(f"  Orientation (rpy): {mae_per_dim[3]:.4f}rad, {mae_per_dim[4]:.4f}rad, {mae_per_dim[5]:.4f}rad")
        avg_pos_mae_mm = np.mean(mae_per_dim[:3]) * 1000
        print(f"\nAverage Position MAE: {avg_pos_mae_mm:.2f} mm")
    else: # Scaled data
         print(f"  Scaled Dims: {mae_per_dim[0]:.4f}, {mae_per_dim[1]:.4f}, {mae_per_dim[2]:.4f}, {mae_per_dim[3]:.4f}, {mae_per_dim[4]:.4f}, {mae_per_dim[5]:.4f}")


    return mae


# --- Plotting Function (No change needed) ---
# Saves to the current directory by default, which is often fine for plots
def plot_losses(train_losses, val_losses):
    """Plots the training and validation loss curves."""
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Training and Validation Losses (Scaled Data)') # Note title change
    plt.xlabel('Epochs')
    plt.ylabel('Loss (MSE)')
    plt.legend()
    plt.grid(True)
    # Consider saving plots also inside MODEL_SAVE_DIR or a dedicated plots dir
    plot_filename = "loss_curves.png"
    plot_save_path = os.path.join(MODEL_SAVE_DIR, plot_filename) # Optional: Save plot in model dir
    # plt.savefig("loss_curves.png") # Original: Saves in current dir
    plt.savefig(plot_save_path) # Saves in MODEL_SAVE_DIR
    print(f"\nLoss curves saved to {plot_save_path}")
    plt.close() # Close the plot to free memory


# --- Main Execution ---
if __name__ == "__main__":
    print("--- Phase 3: Surrogate Model Training (PyTorch with Scaling) ---")

    # --- 1. Load Original Data ---
    print("Loading original data...")
    try:
        X_train_orig = np.load(os.path.join(DATA_DIR, "X_train.npy")).astype(np.float32)
        y_train_orig = np.load(os.path.join(DATA_DIR, "y_train_pose.npy")).astype(np.float32)
        X_test_orig = np.load(os.path.join(DATA_DIR, "X_test.npy")).astype(np.float32)
        y_test_orig = np.load(os.path.join(DATA_DIR, "y_test_pose.npy")).astype(np.float32)
    except FileNotFoundError as e:
        print(f"Error loading data: {e}")
        print(f"Ensure data exists in '{DATA_DIR}' and 'generate_data.py' ran successfully.")
        exit(1)
    except Exception as e:
        print(f"An unexpected error occurred loading data: {e}")
        exit(1)

    # --- 2. Fit and Apply Scalers ---
    print("Fitting and applying data scalers...")
    x_scaler = StandardScaler()
    y_scaler = StandardScaler()

    # Fit SCALERS ONLY on TRAINING data
    X_train_scaled = x_scaler.fit_transform(X_train_orig)
    y_train_scaled = y_scaler.fit_transform(y_train_orig)

    # Transform TEST data using the FITTED scalers
    X_test_scaled = x_scaler.transform(X_test_orig)
    y_test_scaled = y_scaler.transform(y_test_orig)

    # --- 3. Save the Scalers ---
    # Uses the corrected X_SCALER_PATH and Y_SCALER_PATH
    print(f"Saving scalers to {X_SCALER_PATH} and {Y_SCALER_PATH}...")
    try:
        joblib.dump(x_scaler, X_SCALER_PATH)
        joblib.dump(y_scaler, Y_SCALER_PATH)
    except Exception as e:
        print(f"Error saving scalers: {e}")
        # Decide if execution should stop if scalers can't be saved
        # exit(1)


    # --- 4. Create Datasets and Split for Validation (using SCALED data) ---
    # Pass the scaled numpy arrays directly to the Dataset constructor
    try:
        full_train_dataset = RobotPoseDataset(X_train_scaled, y_train_scaled, device=device)
        test_dataset = RobotPoseDataset(X_test_scaled, y_test_scaled, device=device)
    except ValueError as e:
         print(f"Error creating dataset: {e}")
         exit(1)


    num_train_samples = len(full_train_dataset)
    num_val_samples = int(np.floor(VALIDATION_SPLIT * num_train_samples))
    # Ensure validation set is not empty and not larger than train set
    num_val_samples = max(1, min(num_val_samples, num_train_samples - 1)) # Ensure at least 1 sample for val and train
    num_train_subset = num_train_samples - num_val_samples

    if num_train_subset <= 0 or num_val_samples <= 0:
        print(f"Error: Not enough data for train/validation split. Train: {num_train_subset}, Val: {num_val_samples}")
        exit(1)

    # random_split works on Dataset objects
    train_subset, val_subset = random_split(full_train_dataset, [num_train_subset, num_val_samples],
                                            generator=torch.Generator().manual_seed(SEED))

    print(f"Total training samples: {num_train_samples}")
    print(f"Using {len(train_subset)} for training, {len(val_subset)} for validation.")
    print(f"Test samples: {len(test_dataset)}")

    # --- 5. Create DataLoaders ---
    train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True) # drop_last can help with partial batches
    val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # --- 6. Initialize Model, Loss, Optimizer ---
    model = SurrogateNet(INPUT_SIZE, HIDDEN_1, HIDDEN_2, OUTPUT_SIZE, DROPOUT_RATE).to(device)
    print("\nModel Architecture:")
    print(model)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # --- 7. Train Model ---
    # Training function now uses the correct global MODEL_SAVE_PATH implicitly
    train_losses, val_losses = train_model(model, train_loader, val_loader, criterion, optimizer, EPOCHS, device, EARLY_STOPPING_PATIENCE)

    # Plot the loss curves (now saved inside MODEL_SAVE_DIR)
    if train_losses and val_losses: # Only plot if training happened
        plot_losses(train_losses, val_losses)

    # --- 8. Load Best Model and Evaluate ---
    print("\nLoading best model based on validation loss for final evaluation...")
    try:
        # Load the best model saved during training using the corrected MODEL_SAVE_PATH
        best_model = SurrogateNet(INPUT_SIZE, HIDDEN_1, HIDDEN_2, OUTPUT_SIZE, DROPOUT_RATE).to(device)
        # Check if model file exists before loading
        if not os.path.exists(MODEL_SAVE_PATH):
             raise FileNotFoundError(f"Saved model not found at {MODEL_SAVE_PATH}. Was training completed?")
        best_model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=device))
        print("Best model loaded successfully.")
        # Pass the FITTED y_scaler to evaluation for unscaling
        evaluate_model(best_model, test_loader, device, y_scaler) # Pass y_scaler
    except FileNotFoundError as e:
         print(f"Error loading model for evaluation: {e}. Evaluation skipped.")
    except Exception as e:
         print(f"An error occurred during final model loading or evaluation: {e}")


    print("\n--- Surrogate Model Training Complete ---")