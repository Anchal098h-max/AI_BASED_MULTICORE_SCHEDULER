# AI-Based Multicore CPU Scheduling Using LSTM and DQN

## 1. Project Overview

This project aims to develop an AI-based multicore CPU scheduling approach using **Long Short-Term Memory (LSTM)** and **Deep Q-Network (DQN)**.

The LSTM model predicts CPU utilization across 16 CPU cores. The DQN uses these predictions, current CPU utilization, per-core queue lengths, and task characteristics to select a suitable core for an incoming task.

The current implementation focuses on AI model development and scheduling simulation. Integration with the live operating system is a separate project responsibility.

## 2. Project Objectives

- Predict future CPU utilization using LSTM.
- Represent the multicore system as a state for reinforcement learning.
- Select one of 16 CPU cores using DQN.
- Simulate task assignment and execution.
- Store scheduling experiences in a replay buffer.
- Train and evaluate the DQN policy.
- Prepare the AI models for integration with the overall scheduling system.

## 3. Technologies Used

- Python
- PyTorch
- NumPy
- Jupyter Notebook
- Pandas
- Deep Learning
- Deep Reinforcement Learning
- Git and GitHub for version control

## 4. System Architecture

The proposed AI pipeline is:

1. Collect CPU and process-related data.
2. Prepare and preprocess the dataset.
3. Use LSTM to predict utilization for 16 CPU cores.
4. Construct the DQN state.
5. Select a CPU core using the DQN policy.
6. Simulate task assignment and execution.
7. Calculate a reward.
8. Store the transition in the replay buffer.
9. Train and evaluate the DQN model.

```text
CPU and Process Dataset
          |
          v
    Data Preprocessing
          |
          v
     Trained LSTM
          |
          v
Predicted Utilization
     (16 Cores)
          |
          +---------------------+
          |                     |
          v                     v
 Current CPU Utilization   Per-Core Queue Lengths
      (16 values)              (16 values)
          |                     |
          +----------+----------+
                     |
                     v
              Incoming Task
                (4 Features)
                     |
                     v
              DQN State
              (52 Features)
                     |
                     v
              Policy Network
                     |
                     v
          Select Core (0 to 15)
                     |
                     v
          Scheduling Simulation
                     |
                     v
             Reward and Next State
                     |
                     v
               Replay Buffer
                     |
                     v
               DQN Training
```

## 5. LSTM Model

The LSTM model predicts CPU utilization for 16 CPU cores using historical system observations.

### Model configuration

- Input sequence length: 20 timesteps
- Input features: 35
- First LSTM hidden size: 128
- Second LSTM hidden size: 64
- Output size: 16 predicted CPU utilization values

### Input features

The 35 input features consist of:

- 16 per-core CPU utilization values
- 16 per-core queue lengths
- Process count
- Running process count
- Overall CPU utilization

### Output

The model generates predicted utilization values for CPU cores 0 through 15.

The predictions are subsequently used as part of the DQN state.

### Model artifacts

The existing model artifacts are expected to include:

- `cpu_lstm_model.pth`
- `cpu_feature_scaler.pkl`
- `cpu_target_scaler.pkl`

The model and scalers must be loaded consistently with the original training configuration.

## 6. DQN Scheduler

The DQN estimates the value of assigning an incoming task to each available CPU core.

### State representation

The current DQN state contains 52 features.

| Component | Number of features |
|---|---:|
| LSTM-predicted CPU utilization | 16 |
| Current CPU utilization | 16 |
| Per-core queue lengths | 16 |
| Incoming task features | 4 |
| **Total** | **52** |

### Incoming task features

The current implementation uses:

1. Task CPU usage
2. Task runtime
3. Task priority
4. Task waiting time

These features are currently supplied through the task-feature encoding function. The example values used during inference are illustrative values, not verified live task measurements.

### DQN architecture

The network uses fully connected layers:

- Input layer: 52 features
- Hidden layer 1: 128 neurons with ReLU activation
- Hidden layer 2: 128 neurons with ReLU activation
- Output layer: 16 Q-values

Each output corresponds to a possible core assignment.

The policy selects the core with the highest estimated Q-value among eligible cores during greedy inference.

### Training configuration

The current training setup includes:

- Optimizer: Adam
- Learning rate: 0.001
- Loss function: Smooth L1 Loss
- Discount factor: 0.99
- Replay buffer capacity: 10,000 experiences
- Initial synthetic experience collection target: 2,000 tasks
- Epsilon-greedy exploration during experience collection

The target network is initialized as a copy of the policy network. Its parameters should be synchronized according to the chosen DQN training procedure.

## 7. Scheduling Simulation

The current implementation uses simulated per-core queues.

Each simulated task contains fields such as:

- Task ID
- Remaining runtime
- Priority
- Waiting time
- Assigned core

The simulation assigns tasks to selected core queues and advances execution in discrete time steps.

**Limitation:** The current simulation is a simplified queue-based model. It is not an implementation of Linux CFS or EEVDF and does not modify the operating system's real scheduler.

## 8. Replay Buffer

The replay buffer stores transitions for DQN training.

Each experience contains:

- `state`
- `action`
- `reward`
- `next_state`
- `done`
- `next_valid_actions`

The stored state and next state each contain 52 features. The action is a core index from 0 to 15.

### Validation performed

The collected replay buffer was inspected and produced the following results:

| Validation | Observed result |
|---|---|
| Number of experiences | 2,000 |
| State array shape | `(2000, 52)` |
| Action array shape | `(2000,)` |
| Reward array shape | `(2000,)` |
| Next-state array shape | `(2000, 52)` |
| Done array shape | `(2000,)` |
| State values finite | True |
| Next-state values finite | True |
| Reward values finite | True |
| Minimum recorded action | 0 |
| Maximum recorded action | 15 |

The first inspected experience marked all 16 cores as valid actions.

These checks confirm basic data integrity and dimensions. They do not establish that the collected experiences are realistic or that the DQN has learned an effective scheduling policy.

## 9. Current Inference Pipeline

The current inference code constructs task features, updates the environment observation, and selects a core.

```python
live_task_features = encode_task_features(
    task_cpu_usage=25.0,
    task_runtime=5.0,
    task_priority=0,
    task_wait_time_ms=10.0
)

live_state = env.set_observation(
    prediction_utilization,
    current_utilization,
    get_queue_lengths(core_queues),
    live_task_features
)

policy_net.eval()

selected_core = select_action(
    policy_net,
    live_state,
    epsilon=0.0,
    valid_actions=range(NUM_CORES)
)

print("Selected core:", selected_core)
```

This demonstrates the model inference path. The supplied task values are placeholders, and selecting a core in Python does not automatically cause the operating system to execute the task on that core.

## 10. Current Development Status

### Implemented

- Dataset collection and feature preparation
- LSTM model and utilization prediction pipeline
- DQN network architecture
- 52-feature state representation
- Epsilon-greedy core selection
- Simulated task and per-core queue structures
- Replay buffer with next-valid-action support
- Collection of 2,000 synthetic experiences
- Basic replay-buffer shape and finite-value validation

### Still requiring verification or improvement

- Quality and accuracy of LSTM predictions
- Correct loading of the trained DQN checkpoint
- Whether DQN training uses meaningful, changing state transitions
- Reward design based on scheduling outcomes
- Consistent feature normalization
- Handling of task priority, runtime, waiting time, and `vruntime` where appropriate
- Changing CPU utilization and workload conditions during simulation
- DQN training stability and convergence
- Performance comparison against round-robin and least-loaded-core baselines
- Evaluation of waiting time, completion time, throughput, fairness, and load balance

## 11. Important Limitations

The current implementation is an AI scheduling prototype.

The synthetic workload and simplified simulation are useful for development, but they do not reproduce all real operating-system scheduling behavior. The existing reward function primarily estimates load and queue-related costs; it has not yet been demonstrated to optimize real scheduling outcomes.

The LSTM prediction pipeline and DQN inference pipeline are implemented, but their combined scheduling performance still needs systematic evaluation.

## 12. Next Steps

1. Inspect the reward distribution and queue-state variation.
2. Verify the transition and next-valid-action logic.
3. Improve the simulation and reward function.
4. Train the DQN on meaningful scheduling experiences.
5. Evaluate the trained policy on workloads not used for training.
6. Compare its performance with baseline schedulers.
7. Save and verify the final model checkpoints.
8. Hand over the documented model interface and artifacts to the team member responsible for system integration.

## 13. Conclusion

The project has established the initial AI pipeline for multicore CPU scheduling using LSTM and DQN. The LSTM supplies predicted CPU utilization, while the DQN maps the system state to one of 16 possible core assignments.

The next major milestone is to demonstrate that the trained DQN improves scheduling performance under realistic and changing workloads, rather than only confirming that the code executes and produces a core selection.
