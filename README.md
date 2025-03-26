# Cognitive Digital Twin MVP for 6-DOF Robot Arm

This project implements a Minimum Viable Product (MVP) for a cognitive digital twin system controlling a simulated 6-DOF robot arm. It utilizes the PyBullet physics engine for simulation, generates synthetic data, trains a neural network surrogate model to approximate robot kinematics, and employs a simple cognitive controller to drive the simulated robot towards a target pose.

## Project Goal

The primary goal is to demonstrate a pipeline where a learned model (surrogate) is integrated into the control loop of a simulated robot, enabling basic "cognitive" control based on synthetic data. This serves as a foundation for exploring more advanced cognitive robotics concepts.

## Pipeline Overview

The project is structured into several distinct phases, executed sequentially:

1.  **Robot Simulation Setup (`robot.py`):** Defines a Python class (`RobotArm6DOF`) to interface with a 6-DOF robot arm (e.g., UR5) loaded into the PyBullet physics simulator. Provides methods for kinematics, dynamics, and state control.
2.  **Synthetic Data Generation (`generate_data.py`):** Uses the simulation setup (in non-graphical mode) to generate a dataset mapping random joint configurations to corresponding end-effector poses. Synthetic sensor noise is added to mimic real-world imperfections. Data is saved as `.npy` files.
3.  **Surrogate Model Training (`train_surrogate.py`):** Loads the synthetic data, preprocesses it (scaling), defines a PyTorch neural network (`SurrogateNet`), and trains it to learn the mapping from joint angles to end-effector pose. The trained model (`.pth`) and data scalers (`.gz`) are saved.
4.  **Cognitive Layer (`cognitive_controller.py`):** Implements a `CognitiveController` class that loads the trained surrogate model and scalers. It uses the model to predict the current pose based on joint angles, calculates the error relative to a target pose, and computes a PI-like correction signal. **(Note: Includes fix for data scaling)**.
5.  **Integrated Simulation (`run_simulation.py`):** Integrates the simulation, cognitive controller, and PyBullet's GUI. It runs a closed loop where the controller calculates corrections, which are then applied (currently via simple addition to target joint angles) to the simulated robot using PyBullet's position controller. Visualizes the target and the robot's end-effector path.
6.  **Validation (`validate_model.py`):** Evaluates the performance of the trained surrogate model on a separate test dataset. Calculates metrics like Mean Absolute Error (MAE) overall, per dimension (position/orientation), and checks temporal consistency.

## Tech Stack / Dependencies

*   Python (3.8+)
*   PyBullet: Physics simulation and visualization.
*   NumPy: Numerical operations.
*   SciPy: Used for spatial transformations (e.g., `Rotation`).
*   PyTorch: Neural network framework for the surrogate model.
*   Scikit-learn: For data preprocessing (`StandardScaler`).
*   Joblib: For saving/loading scikit-learn scalers.
*   Matplotlib: For plotting training/validation loss curves.
*   `robot_descriptions`: Helper library to easily load common robot URDFs (like UR5).

## Setup

1.  **Clone the repository:**
    ```bash
    git clone <your-repo-url>
    cd <your-repo-name>
    ```

2.  **Create and activate a virtual environment (recommended):**
    ```bash
    python -m venv venv
    # On Windows
    .\venv\Scripts\activate
    # On macOS/Linux
    source venv/bin/activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install pybullet numpy scipy torch scikit-learn joblib matplotlib robot_descriptions
    # Or, preferably, if a requirements.txt file is created:
    # pip install -r requirements.txt
    ```
    *(Note: Ensure PyTorch installation command matches your system/CUDA setup if using GPU - see official PyTorch website)*

## Usage

The scripts should generally be run in the following order, as they generate outputs needed by subsequent steps:

1.  **Generate Synthetic Data:**
    ```bash
    python generate_data.py
    ```
    *   This creates the `synthetic_data/` directory with `X_train.npy`, `y_train_pose.npy`, `X_test.npy`, `y_test_pose.npy`.

2.  **Train the Surrogate Model:**
    ```bash
    python train_surrogate.py
    ```
    *   This creates the `surrogate_models/` directory with `surrogate_model.pth`, `x_scaler.gz`, `y_scaler.gz`, and `loss_curves.png`.

3.  **Run the Integrated Simulation (with GUI):**
    ```bash
    python run_simulation.py
    ```
    *   This opens a PyBullet window showing the robot attempting to reach the hardcoded target pose using the cognitive controller. Press `Ctrl+C` in the terminal to stop.

4.  **Validate the Trained Model:**
    ```bash
    python validate_model.py
    ```
    *   This loads the test data and the trained model/scalers, then prints performance metrics (MAE, temporal consistency) to the console.

## Project Structure

```
.
├── robot.py # PyBullet robot interface class
├── generate_data.py # Script for synthetic data generation
├── train_surrogate.py # Script for training the surrogate NN model
├── cognitive_controller.py # Cognitive controller class using the surrogate
├── run_simulation.py # Script for running the integrated simulation loop
├── validate_model.py # Script for evaluating the surrogate model
├── .gitignore # Specifies intentionally untracked files
├── README.md # This file
│
├── synthetic_data/ # (Generated) Contains .npy datasets
│ ├── X_train.npy
│ ├── y_train_pose.npy
│ ├── X_test.npy
│ └── y_test_pose.npy
│
└── surrogate_models/ # (Generated) Contains trained model and scalers
├── surrogate_model.pth
├── x_scaler.gz
├── y_scaler.gz
└── loss_curves.png # (Generated plot)
```

## Current Status & Limitations

*   This is an **MVP** demonstrating the core pipeline concept.
*   The control strategy (`run_simulation.py`) is currently very basic (adding task-space correction directly to joint angles), which is kinematically/dynamically naive and likely suboptimal/unstable.
*   Relies purely on synthetic data; no real-world data interaction.
*   The surrogate model is a simple MLP and doesn't capture uncertainty.
*   Error handling and parameter configuration are minimal.

## TODO List / Future Goals

Here are potential improvements and directions for future work:

**Control Strategy & Integration:**

*   [ ] **Implement Jacobian-Based Control:** Use the robot's Jacobian (via `p.calculateJacobian`) and the pose error to command joint velocities (`p.VELOCITY_CONTROL`), respecting kinematics.
*   [ ] **Implement Operational Space Control (OSC):** Develop a controller that computes joint torques (`p.TORQUE_CONTROL`) based on task-space dynamics.
*   [ ] **Implement IK-Based Correction:** Use the pose error to slightly adjust the target pose, then use `p.calculateInverseKinematics` to find target joint angles for `p.POSITION_CONTROL`.
*   [ ] **Explore Model Predictive Control (MPC):** Use the surrogate model within an MPC framework for optimized control over a horizon.

**Surrogate Model Enhancements:**

*   [ ] **Uncertainty Quantification:** Replace MLP with a Bayesian Neural Network (BNN) or Deep Ensemble to estimate model uncertainty. Use uncertainty in the control loop (e.g., adaptive gains, safety limits).
*   [ ] **Improved Orientation Representation/Loss:** Use rotation matrices or quaternions directly with appropriate geometric loss functions instead of Euler angles (which have singularity/wrapping issues).
*   [ ] **Physics-Informed Neural Networks (PINN):** Incorporate kinematic constraints into the loss function.
*   [ ] **Architecture Exploration:** Experiment with different NN architectures (e.g., attention mechanisms).
*   [ ] **Dynamic Model:** Train surrogate to predict future states or required torques (inverse dynamics).

**Data Generation:**

*   [ ] **Improved Sampling:** Use quasi-random sequences (Halton, Sobol) or workspace-aware sampling instead of uniform random.
*   [ ] **More Realistic Noise:** Implement more complex sensor noise models (correlated, bias, etc.).
*   [ ] **Dynamic Data:** Generate trajectories including velocities and accelerations.
*   [ ] **Domain Randomization:** Randomize physics parameters (mass, friction, etc.) and sensor noise during data generation to improve robustness (Sim2Real).
*   [ ] **Collision Data:** Generate data near collisions to potentially train a collision prediction model.

**Simulation & Realism:**

*   [ ] **Tune Physics Parameters:** Identify and set more realistic joint friction, damping, and inertia in the URDF or via PyBullet API.
*   [ ] **Actuator Modeling:** Implement more realistic motor models (dynamics, delays, torque limits).

**Validation & Testing:**

*   [ ] **More Metrics:** Report RMSE, max error, R-squared, dedicated orientation error metrics.
*   [ ] **Closed-Loop Evaluation:** Test the *integrated system* on predefined trajectories (lines, circles) and measure tracking performance (error, settling time, stability).
*   [ ] **Robustness Tests:** Evaluate performance with simulated disturbances or randomized parameters.

**Software Engineering:**

*   [ ] **Configuration Management:** Use YAML files (e.g., via Hydra) to manage parameters instead of hardcoding them.
*   [ ] **Create `requirements.txt`:** Automatically generate from the virtual environment (`pip freeze > requirements.txt`).
*   [ ] **ROS/ROS2 Integration:** Refactor components into ROS/ROS2 nodes for standardization and easier hardware deployment.
*   [ ] **Modularization:** Structure code into a more formal Python package.
*   [ ] **Unit/Integration Tests:** Add tests for individual components and their interactions.