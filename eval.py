import numpy as np
import os
import matplotlib.pyplot  as plt
path = "bayes_opt_runs/"

cnn = np.load(path+"workload_cnn_1.npz", allow_pickle=True)
transformer = np.load(path+"workload_cnn_1.npz", allow_pickle=True)
transfer = np.load(path+"workload_cnn_1_workload_transformer_1.npz", allow_pickle=True)

y_cnn = cnn["y"]
y_transformer = transformer["y"]
y_transfer = transfer["y"]


best_cnn = np.minimum.accumulate(y_cnn)
best_transformer = np.minimum.accumulate(y_transformer)
best_transfer = np.minimum.accumulate(y_transfer)

plt.figure(figsize=(9, 6))
plt.plot(
    np.arange(1, len(y_cnn) + 1),
    best_cnn,
    label="CNN-1"
)
plt.plot(
    np.arange(1, len(y_transformer) + 1),
    best_transformer,
    label="Transformer-1"
)
plt.plot(
    np.arange(1, len(y_transfer) + 1),
    best_transfer,
    label="CNN → Transformer-1"
)
plt.xlabel("Number of CarbonPATH evaluations")
plt.ylabel("Best cost found so far")
plt.title("BO convergence")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

baseline_initial_best = np.min(y_transformer[:200])
baseline_final_best = np.min(y_transformer)

transfer_initial_best = np.min(y_transfer[:200])
transfer_final_best = np.min(y_transfer)

print("Transformer from scratch")
print("------------------------")
print("Best after 200 initial evaluations:",
      baseline_initial_best)
print("Final best:",
      baseline_final_best)

print()

print("CNN -> Transformer transfer")
print("---------------------------")
print("Best after 200 initial evaluations:",
      transfer_initial_best)
print("Final best:",
      transfer_final_best)

print()

print("Transfer improvement:")
print(
    "Absolute:",
    baseline_final_best - transfer_final_best
)

print(
    "Percentage:",
    100 * (baseline_final_best - transfer_final_best)
    / baseline_final_best
)