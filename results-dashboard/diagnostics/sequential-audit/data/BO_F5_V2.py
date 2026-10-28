# -*- coding: utf-8 -*-
"""
Created on Fri Jul 24 20:50:53 2026

@author: ariel
"""

import warnings
import numpy as np
import matplotlib.pyplot as plt

from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    ConstantKernel,
    RBF,
    WhiteKernel
)

warnings.filterwarnings("ignore", category=ConvergenceWarning)


# ============================================================
# 1. Load the original Function 5 data
# ============================================================

X_initial = np.load("initial_inputs.npy")
y_initial = np.load("initial_outputs.npy").reshape(-1)

print("Original input shape:", X_initial.shape)
print("Original output shape:", y_initial.shape)


# ============================================================
# 2. Add the newest real process evaluation
# ============================================================

x_new = np.array([
    [0.848524, 0.780348, 0.997475, 0.584310],
    [0.551534, 0.678167, 0.998240, 0.011006],
    [0.138355, 0.799538, 0.996283, 0.985984],
    [0.081435, 0.002022, 0.102854, 0.993462],
    [0.816264, 0.691860, 0.979595, 0.999792],
    [0.797875, 0.744728, 0.819544, 0.998029],
    [0.716113, 0.000906, 0.920346, 0.988451],
    [0.000098, 0.379683, 0.985449, 0.988715],
    [0.006955, 0.280719, 0.004237, 0.017733],
    [0.975542, 0.988882, 0.999175, 0.871281],
    [0.977955, 0.981488, 0.994672, 0.406285],
    [0.001580, 0.545966, 0.421144, 0.996278]
])

y_new = np.array([
    1647.34607390959,
    480.65995893854,
    2438.19340978837,
    172.77228013589,
    3165.58148992445,
    2013.91447357049,
    1504.35293721349,
    1457.33570927710,
    157.65963514479,
    6136.65146651652,
    3941.67731214184,
    281.09099906932
])

X = np.vstack([X_initial, x_new])
y = np.concatenate([y_initial, y_new])

print("Updated input shape:", X.shape)
print("Updated output shape:", y.shape)


# ============================================================
# 3. Transform the yield
# ============================================================

if np.any(y <= 0):
    raise ValueError(
        "All yield values must be positive because log(y) is used."
    )

# Maximising log(y) is equivalent to maximising y.
y_model = np.log(y)


# ============================================================
# 4. Define and fit the Gaussian Process
# ============================================================

kernel = (
    ConstantKernel(
        constant_value=1.0,
        constant_value_bounds=(1e-3, 1e3)
    )
    *
    RBF(
        length_scale=[0.25, 0.25, 0.25, 0.25],
        length_scale_bounds=(1e-2, 5.0)
    )
    +
    WhiteKernel(
        noise_level=1e-6,
        noise_level_bounds=(1e-10, 1e-1)
    )
)

gp = GaussianProcessRegressor(
    kernel=kernel,
    normalize_y=True,
    n_restarts_optimizer=10,
    random_state=4
)

gp.fit(X, y_model)

print("\nOptimised GP kernel:")
print(gp.kernel_)


# ============================================================
# 5. Generate candidate process conditions
# ============================================================

rng = np.random.default_rng(4)

number_of_candidates = 300_000

lower_bounds = np.zeros(4)
upper_bounds = np.ones(4)

X_candidates = rng.uniform(
    low=lower_bounds,
    high=upper_bounds,
    size=(number_of_candidates, 4)
)


# ============================================================
# 6. Predict posterior mean and uncertainty
# ============================================================

posterior_mean_log, posterior_std_log = gp.predict(
    X_candidates,
    return_std=True
)

# Convert the GP posterior mean back to the yield scale.
posterior_yield = np.exp(posterior_mean_log)


# ============================================================
# 7. UCB acquisition function
# ============================================================

# For maximisation:
# UCB(x) = mean_log(x) + kappa * std_log(x)

kappa = 1.96

acquisition = (
    posterior_mean_log
    + kappa * posterior_std_log
)


# ============================================================
# 8. Avoid sampling too close to an existing condition
# ============================================================

minimum_distance = 0.01

distances = np.linalg.norm(
    X_candidates[:, None, :] - X[None, :, :],
    axis=2
)

closest_distance = distances.min(axis=1)

acquisition[
    closest_distance < minimum_distance
] = -np.inf


# ============================================================
# 9. Select the next UCB sampling point
# ============================================================

next_index = np.argmax(acquisition)

next_point = X_candidates[next_index]
next_predicted_yield = posterior_yield[next_index]
next_predicted_std_log = posterior_std_log[next_index]


# ============================================================
# 10. Identify the best observed point
# ============================================================

best_observed_index = np.argmax(y)

best_observed_point = X[best_observed_index]
best_observed_yield = y[best_observed_index]


# ============================================================
# 11. Identify the maximum GP posterior mean
# ============================================================

best_mean_index = np.argmax(posterior_mean_log)

best_surrogate_point = X_candidates[best_mean_index]
best_surrogate_yield = posterior_yield[best_mean_index]
best_surrogate_std_log = posterior_std_log[best_mean_index]


# ============================================================
# 12. Print results
# ============================================================

print("\n" + "=" * 74)
print("NEW REAL PROCESS EVALUATION")
print("=" * 74)

print(
    f"x1 = {x_new[0, 0]:.6f}, "
    f"x2 = {x_new[0, 1]:.6f}, "
    f"x3 = {x_new[0, 2]:.6f}, "
    f"x4 = {x_new[0, 3]:.6f}"
)

print(
    f"Real yield = "
    f"{y_new[0]:.8f}"
)


print("\n" + "=" * 74)
print("BEST OBSERVED PROCESS CONDITION SO FAR")
print("=" * 74)

print(
    f"x1 = {best_observed_point[0]:.6f}, "
    f"x2 = {best_observed_point[1]:.6f}, "
    f"x3 = {best_observed_point[2]:.6f}, "
    f"x4 = {best_observed_point[3]:.6f}"
)

print(
    f"Observed yield = "
    f"{best_observed_yield:.8f}"
)


print("\n" + "=" * 74)
print("MAXIMUM GP POSTERIOR MEAN")
print("=" * 74)

print(
    f"x1 = {best_surrogate_point[0]:.6f}, "
    f"x2 = {best_surrogate_point[1]:.6f}, "
    f"x3 = {best_surrogate_point[2]:.6f}, "
    f"x4 = {best_surrogate_point[3]:.6f}"
)

print(
    f"Predicted yield = "
    f"{best_surrogate_yield:.8f}"
)

print(
    f"Posterior standard deviation on log scale = "
    f"{best_surrogate_std_log:.8f}"
)


print("\n" + "=" * 74)
print("SUGGESTED NEXT SAMPLING POINT USING UCB")
print("=" * 74)

print(
    f"x1 = {next_point[0]:.6f}, "
    f"x2 = {next_point[1]:.6f}, "
    f"x3 = {next_point[2]:.6f}, "
    f"x4 = {next_point[3]:.6f}"
)

print(
    f"Predicted yield = "
    f"{next_predicted_yield:.8f}"
)

print(
    f"Posterior standard deviation on log scale = "
    f"{next_predicted_std_log:.8f}"
)

print(f"UCB exploration parameter = {kappa}")


# ============================================================
# 13. Compare observed and GP-fitted yields
# ============================================================

training_mean_log, training_std_log = gp.predict(
    X,
    return_std=True
)

training_predicted_yield = np.exp(training_mean_log)

plt.figure(figsize=(8, 6))

plt.scatter(
    y,
    training_predicted_yield,
    s=85,
    label="Evaluated conditions"
)

plot_min = min(y.min(), training_predicted_yield.min())
plot_max = max(y.max(), training_predicted_yield.max())

plt.plot(
    [plot_min, plot_max],
    [plot_min, plot_max],
    linestyle="--",
    label="Perfect prediction"
)

plt.xscale("log")
plt.yscale("log")
plt.xlabel("Observed yield")
plt.ylabel("GP-predicted yield")
plt.title("Observed versus predicted yields — Function 5")
plt.legend()
plt.tight_layout()
plt.show()


# ============================================================
# 14. Pairwise posterior projection
# ============================================================

# Vary x2 and x3 while fixing x1 and x4 at the
# surrogate-predicted best condition.

grid_size = 120

x2_values = np.linspace(0, 1, grid_size)
x3_values = np.linspace(0, 1, grid_size)

X2_grid, X3_grid = np.meshgrid(
    x2_values,
    x3_values
)

projection_candidates = np.column_stack(
    [
        np.full(X2_grid.size, best_surrogate_point[0]),
        X2_grid.ravel(),
        X3_grid.ravel(),
        np.full(X2_grid.size, best_surrogate_point[3])
    ]
)

projection_mean_log = gp.predict(
    projection_candidates
)

projection_yield = np.exp(
    projection_mean_log
).reshape(grid_size, grid_size)

plt.figure(figsize=(9, 7))

contour = plt.contourf(
    X2_grid,
    X3_grid,
    projection_yield,
    levels=30
)

plt.colorbar(
    contour,
    label="Predicted yield"
)

plt.scatter(
    X_initial[:, 1],
    X_initial[:, 2],
    marker="x",
    s=75,
    label="Initial conditions"
)

plt.scatter(
    x_new[0, 1],
    x_new[0, 2],
    marker="o",
    s=150,
    facecolors="none",
    linewidths=2,
    label="Newest real evaluation"
)

plt.scatter(
    next_point[1],
    next_point[2],
    marker="*",
    s=240,
    label="Suggested next condition"
)

plt.xlabel("x2")
plt.ylabel("x3")

plt.title(
    "Updated GP yield projection — Function 5\n"
    f"x1 = {best_surrogate_point[0]:.3f}, "
    f"x4 = {best_surrogate_point[3]:.3f}"
)

plt.legend()
plt.tight_layout()
plt.show()