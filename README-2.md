# Live CPU Data Collection, Process Queue Tracking, and LSTM Utilization Prediction

## 1. Project Overview

This module collects live CPU and process data from a Linux system, tracks the **running queue** and **ready queue** of programs on each CPU, and uses a trained **Long Short-Term Memory (LSTM)** model to predict CPU utilization across 16 logical CPU cores.

It is the live-data component of the AI-Based Multicore CPU Scheduling project (LSTM and DQN). The predictions produced here are intended to feed the DQN state, as described in the main project README.

The script is a monitoring and prediction tool. It does not modify the operating system's scheduler and does not assign tasks to cores.

## 2. Objectives

- Sample live per-core CPU utilization once per second.
- Read the per-core scheduler queue length (`nr_running`) from the Linux kernel.
- Count processes and runnable processes system-wide.
- Maintain a per-CPU running queue and ready queue of programs.
- Build the 35-feature input vector expected by the trained LSTM.
- Predict utilization for CPU cores 0 through 15 from a sliding window of 20 samples.
- Print each prediction together with the current queue state.

## 3. Technologies Used

- Python
- PyTorch
- NumPy
- psutil
- joblib (scaler loading)
- Linux scheduler debug interface (`/sys/kernel/debug/sched/debug` or `/proc/sched_debug`)

## 4. System Architecture

The runtime pipeline is:

1. Validate that the model file, both scalers, and the 16-CPU configuration are available.
2. Sample per-core CPU utilization for one second.
3. Refresh the process queues and process counts.
4. Read per-core queue lengths from the kernel scheduler data.
5. Assemble the 35-feature vector.
6. Append it to a rolling window of 20 samples.
7. Scale the window, run the LSTM, and inverse-scale the output.
8. Print the predicted utilization and the running/ready queues.

```text
        psutil.cpu_percent (1 s, per core)
                      |
                      v
       CPU utilization (16 values) --------------------+
                                                       |
   psutil.process_iter                                 |
          |                                            |
          v                                            |
   ProcessQueueTracker.update()                        |
     |            |                                    |
     v            v                                    |
 Process count   Running + ready count                 |
     |            |                                    |
     |            |      Kernel scheduler debug file   |
     |            |        (nr_running per CPU)        |
     |            |                |                   |
     |            |                v                   |
     |            |      Queue lengths (16 values)     |
     |            |                |                   |
     +------------+----------------+-------------------+
                      |
                      v
          Feature Vector (35 features)
                      |
                      v
        Rolling Window (20 x 35 samples)
                      |
                      v
          Feature Scaler (transform)
                      |
                      v
                Trained LSTM
                      |
                      v
          Target Scaler (inverse)
                      |
                      v
   Predicted Utilization (16 cores) + Queue Report
```

## 5. LSTM Model

The model is defined by the `CpuUtilizationLSTM` class and uses the same configuration as the training notebook.

### Model configuration

- Input sequence length: 20 timesteps
- Input features: 35
- First LSTM hidden size: 128
- Second LSTM hidden size: 64
- Dropout: 0
- Output size: 16 predicted CPU utilization values

The output of the final timestep of the second LSTM passes through a fully connected layer to produce the 16 predictions.

### Input features

The 35 input features are, in this exact order:

| Index | Feature | Source |
|---|---|---|
| 0–15 | Per-core CPU utilization | `psutil.cpu_percent(percpu=True)` |
| 16–31 | Per-core queue length | `nr_running` from the scheduler debug file |
| 32 | Process count | Number of processes seen by the tracker |
| 33 | Running process count | Number of processes in state `R` (running + ready) |
| 34 | Overall CPU utilization | Mean of the 16 per-core values |

The feature order must match the training notebook. The collected vector is rejected if it is not exactly 35 values or contains non-finite values.

### Output

The model generates predicted utilization values for CPU cores 0 through 15, converted back to the original scale using the target scaler.

### Model artifacts

The following files must be present in the working directory:

- `cpu_lstm_model.pth`
- `cpu_feature_scaler.pkl`
- `cpu_target_scaler.pkl`

At startup the script also checks that the feature scaler expects 35 features and the target scaler expects 16 outputs.

## 6. Process Queue Tracker

The `ProcessQueueTracker` class maintains a **running queue** and a **ready queue** for each CPU.

### How it works

On Linux, a process in state `R` is either executing on a CPU or waiting on that CPU's run queue. User space cannot tell which directly, so the tracker approximates it:

1. For each process in state `R`, it records the CPU it is on (`cpu_num`).
2. It computes the CPU time the process consumed since the previous sample.
3. On each CPU, the `R` process with the largest CPU-time delta is treated as **running**.
4. The remaining `R` processes on that CPU are treated as **ready** (waiting), ordered by CPU-time delta from highest to lowest.

### Data structures

- `running_queue[cpu]`: list of at most one process entry
- `ready_queue[cpu]`: list of waiting process entries
- Each entry: `{"pid": int, "name": str, "cpu_delta": float}`

### Main members

| Member | Purpose |
|---|---|
| `update()` | Rescans all processes and rebuilds both queues, the process count, and the CPU-time baseline |
| `running_count` | Total processes in the running queues across all CPUs |
| `ready_count` | Total processes in the ready queues across all CPUs |
| `snapshot()` | Returns a copy of the queues that is safe to store in a history list |
| `format_queues()` | Returns a printable per-CPU summary |

Processes that exit or become inaccessible during a scan are skipped.

## 7. Feature Collection

`collect_features(tracker)` builds one 35-feature sample per call:

1. Waits one second and reads per-core CPU utilization.
2. Verifies that the device reports exactly 16 logical CPUs.
3. Computes overall CPU utilization as the mean of the per-core values.
4. Calls `tracker.update()` to refresh the queues, process count, and running process count.
5. Calls `read_per_cpu_queues()` to read the kernel queue length for each CPU.
6. Concatenates the values into a `float32` vector.

### Kernel queue reader

`read_per_cpu_queues()` parses the first available file from:

- `/sys/kernel/debug/sched/debug`
- `/proc/sched_debug`

It reads the `cpu#N` section headers and the `nr_running` value inside each section. An error is raised if neither file exists, the file cannot be read, or any CPU from 0 to 15 has no reading.

## 8. Runtime Behavior

### Startup checks

- All three model artifacts exist.
- The system reports exactly 16 logical CPUs.
- The feature scaler expects 35 features and the target scaler expects 16.
- The model weights load successfully (with a compatibility fallback for older PyTorch versions).

The script uses a GPU if one is available and otherwise runs on the CPU.

### Prediction loop

- One sample is collected per loop, taking about one second.
- The first prediction appears after 20 samples have been collected.
- After that, a prediction is made on every new sample using the most recent 20 samples.
- A parallel history of queue snapshots is kept, aligned with the feature history.

### Example output

```text
2025-01-01 12:00:00 Logical CPU core: 5 Predicted CPU utilization: [12.34, 8.1, ..., 3.27]
  Running queue: 4 | Ready queue: 2
  CPU  0 | running: python3(1234) | ready: [chrome(5678)]
  CPU  3 | running: code(4321) | ready: -
```

The printed values are illustrative. Each line lists the 16 predicted utilization values in core order.

### Logical CPU core label

The "Logical CPU core" shown in each output line is the core on which this Python process itself most recently ran, read with `cpu_num()` just before printing. The process can migrate between cores, so this is a point-in-time reading and is not a scheduling decision.

## 9. Requirements and Usage

### Requirements

- Linux with a scheduler debug interface available
- Read access to `/sys/kernel/debug/sched/debug` or `/proc/sched_debug`, which typically requires root or a mounted debugfs
- A machine with exactly 16 logical CPUs
- Python packages: `torch`, `numpy`, `psutil`, `joblib`
- The three model artifacts listed in Section 5

### Run

```bash
sudo python3 final_queue_live_tracker.py
```

Replace the file name with the actual name of the script. Press `Ctrl+C` to stop; the script prints `Stopped.` and exits.

### Configuration constants

| Constant | Value | Meaning |
|---|---|---|
| `N_CPUS` | 16 | Required logical CPU count |
| `WINDOW_SIZE` | 20 | Samples per LSTM input window |
| `POLL_SECONDS` | 1.0 | CPU utilization sampling interval |

## 10. Current Development Status

### Implemented

- LSTM model definition matching the training configuration
- Live per-core CPU utilization sampling
- Per-core queue length reading from the Linux scheduler data
- Per-CPU running and ready queue tracking
- Rolling 20-sample window with scaling and inverse scaling
- Startup validation of artifacts, scaler dimensions, and CPU count
- Live prediction printing with queue reports

### Still requiring verification or improvement

- Accuracy of the live LSTM predictions compared with measured utilization
- Accuracy of the running/ready classification against the real kernel state
- Consistency between live feature values and the training data distribution
- Behavior on systems with a different number of CPUs
- Passing the predictions to the DQN scheduler through a documented interface

## 11. Important Limitations

- **Queue classification is an approximation.** User space cannot see which `R` process is actually on the CPU. The "running" process is inferred as the one that used the most CPU time since the last sample, so it can be wrong, especially at short intervals.
- **First-sample deltas are zero.** A process seen for the first time has no previous CPU time, so its delta is 0 on its first appearance.
- **Fixed to 16 CPUs.** The script stops with an error if the logical CPU count is not 16.
- **Requires privileged access.** Without access to the scheduler debug file, the script cannot start collecting features.
- **Kernel-dependent file location.** Newer kernels expose the scheduler data under debugfs rather than `/proc/sched_debug`; the script checks both.
- **Tracker output is informational.** The running and ready queues are printed for inspection. Only the process count, the running-plus-ready count, and the kernel `nr_running` values enter the model input.
- **Prediction only.** The script does not change which core any process runs on.
- **Sampling timing.** CPU utilization is sampled over one second, and the process scan and kernel queue read occur immediately afterward, so the features within one sample are not captured at exactly the same instant.

## 12. Next Steps

1. Compare live predictions with the utilization actually observed in the following seconds.
2. Validate the running/ready approximation against kernel scheduler data.
3. Confirm that live feature ranges match the training dataset.
4. Provide the predictions to the DQN state through a stable interface.
5. Make the CPU count configurable instead of fixed at 16, if retraining allows it.
6. Document the model interface and artifacts for the team member responsible for system integration.

## 13. Conclusion

This module provides the live data path for the project: it gathers per-core utilization and kernel queue lengths, assembles the 35-feature input, and runs the trained LSTM to predict utilization for 16 cores, while also reporting a per-CPU running and ready queue of programs. The next milestone is to verify the prediction accuracy and queue-tracking approximation on live workloads and to connect the output to the DQN scheduler.
