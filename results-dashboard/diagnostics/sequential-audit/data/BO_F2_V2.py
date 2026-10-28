# -*- coding: utf-8 -*-
"""
Created on Fri Jul 24 17:53:40 2026

@author: ariel
"""

import warnings
import numpy as np
import matplotlib.pyplot as plt

from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    ConstantKernel,
    Matern,
    WhiteKernel
)
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=ConvergenceWarning)


# ============================================================
# 1. Load the original Function 2 observations
# ============================================================

X_initial = np.load("initial_inputs.npy")
y_initial = np.load("initial_outputs.npy").reshape(-1)

print("Original input shape:", X_initial.shape)
print("Original output shape:", y_initial.shape)


# ============================================================
# 2. Add the new real evaluation
# ============================================================

x_new = np.array([[0.736667, 0.000000],
                  [0.96, 1.0],
                  [0.693333, 0.863333],
                  [0.696667, 0.000000],
                  [0.0, 1.0],
                  [0.693333, 0.0],
                  [0.690000, 1.000000],
                  [0.700000, 0.000000],
                  [0.703333, 0.000000],
                  [0.690000, 0.000000],
                  [0.690000, 0.003333],
                  [0.693333, 0.003333]
                  ])
y_new = np.array([0.420349034063801,
                  0.020627119001356,
                  0.608625657233808,
                  0.664601456437472,
                  0.063592899494264,
                  0.564626094993950,
                  0.563604154985633,
                  0.697183249548196,
                  0.537834253232825,
                  0.644646072973282,
                  0.617678437432567,
                  0.697245804354649
                  ])

X = np.vstack([X_initial, x_new])
y = np.concatenate([y_initial, y_new])

print("Updated input shape:", X.shape)
print("Updated output shape:", y.shape)


# ============================================================
# 3. Standardise the objective
# ============================================================

y_scaler = StandardScaler()

y_scaled = y_scaler.fit_transform(
    y.reshape(-1, 1)
).ravel()


# ============================================================
# 4. Define and fit the GP surrogate
# ============================================================

kernel = (
    ConstantKernel(
        constant_value=1.0,
        constant_value_bounds=(1e-3, 1e3)
    )
    *
    Matern(
        length_scale=[0.2, 0.2],
        length_scale_bounds=(1e-2, 5.0),
        nu=2.5
    )
    +
    WhiteKernel(
        noise_level=1e-3,
        noise_level_bounds=(1e-8, 1.0)
    )
)

gp = GaussianProcessRegressor(
    kernel=kernel,
    alpha=1e-8,
    normalize_y=False,
    n_restarts_optimizer=15,
    random_state=4
)

gp.fit(X, y_scaled)

print("\nOptimised GP kernel:")
print(gp.kernel_)


# ============================================================
# 5. Create the two-dimensional candidate grid
# ============================================================

grid_size = 301

x1_grid = np.linspace(0, 1, grid_size)
x2_grid = np.linspace(0, 1, grid_size)

X1_grid, X2_grid = np.meshgrid(
    x1_grid,
    x2_grid
)

X_candidates = np.column_stack(
    [
        X1_grid.ravel(),
        X2_grid.ravel()
    ]
)


# ============================================================
# 6. Calculate posterior mean and uncertainty
# ============================================================

posterior_mean_scaled, posterior_std_scaled = gp.predict(
    X_candidates,
    return_std=True
)

posterior_mean = y_scaler.inverse_transform(
    posterior_mean_scaled.reshape(-1, 1)
).ravel()

posterior_std = (
    posterior_std_scaled
    * y_scaler.scale_[0]
)


# ============================================================
# 7. Upper Confidence Bound acquisition
# ============================================================

kappa = 1.96

acquisition = (
    posterior_mean_scaled
    + kappa * posterior_std_scaled
)


# ============================================================
# 8. Avoid querying an already evaluated neighbourhood
# ============================================================

minimum_distance = 0.01

distances = np.sqrt(
    (
        X_candidates[:, None, :]
        - X[None, :, :]
    ) ** 2
).sum(axis=2) ** 0.5

closest_distance = distances.min(axis=1)

acquisition[
    closest_distance < minimum_distance
] = -np.inf


# ============================================================
# 9. Select the next query point
# ============================================================

next_index = np.argmax(acquisition)

next_point = X_candidates[next_index]
next_predicted_mean = posterior_mean[next_index]
next_predicted_std = posterior_std[next_index]


# ============================================================
# 10. Current best observed point
# ============================================================

best_observed_index = np.argmax(y)

best_observed_point = X[best_observed_index]
best_observed_value = y[best_observed_index]


# ============================================================
# 11. Point with the highest GP posterior mean
# ============================================================

best_mean_index = np.argmax(posterior_mean)

best_surrogate_point = X_candidates[best_mean_index]
best_surrogate_mean = posterior_mean[best_mean_index]
best_surrogate_std = posterior_std[best_mean_index]


# ============================================================
# 12. Print the results
# ============================================================

print("\n" + "=" * 68)
print("NEW REAL OBSERVATION")
print("=" * 68)

print(
    f"x1 = {x_new[0, 0]:.6f}, "
    f"x2 = {x_new[0, 1]:.6f}"
)

print(f"Real objective value = {y_new[0]:.12f}")


print("\n" + "=" * 68)
print("BEST OBSERVED POINT SO FAR")
print("=" * 68)

print(
    f"x1 = {best_observed_point[0]:.6f}, "
    f"x2 = {best_observed_point[1]:.6f}"
)

print(f"Observed objective = {best_observed_value:.12f}")


print("\n" + "=" * 68)
print("MAXIMUM GP POSTERIOR MEAN")
print("=" * 68)

print(
    f"x1 = {best_surrogate_point[0]:.6f}, "
    f"x2 = {best_surrogate_point[1]:.6f}"
)

print(f"Predicted mean = {best_surrogate_mean:.12f}")
print(f"Predicted standard deviation = {best_surrogate_std:.12f}")


print("\n" + "=" * 68)
print("SUGGESTED NEXT SAMPLING POINT USING UCB")
print("=" * 68)

print(
    f"x1 = {next_point[0]:.6f}, "
    f"x2 = {next_point[1]:.6f}"
)

print(f"Predicted mean = {next_predicted_mean:.12f}")
print(f"Predicted standard deviation = {next_predicted_std:.12f}")
print(f"UCB exploration parameter = {kappa}")


# ============================================================
# 13. Posterior-mean plot
# ============================================================

mean_surface = posterior_mean.reshape(
    grid_size,
    grid_size
)

plt.figure(figsize=(10, 7))

contour = plt.contourf(
    X1_grid,
    X2_grid,
    mean_surface,
    levels=30
)

plt.colorbar(
    contour,
    label="GP posterior mean"
)

plt.scatter(
    X_initial[:, 0],
    X_initial[:, 1],
    marker="x",
    s=90,
    label="Initial observations"
)

plt.scatter(
    x_new[0, 0],
    x_new[0, 1],
    marker="o",
    s=150,
    facecolors="none",
    linewidths=2,
    label="Newest real evaluation"
)

plt.scatter(
    next_point[0],
    next_point[1],
    marker="*",
    s=250,
    label="Suggested next point"
)

plt.xlabel("x1")
plt.ylabel("x2")
plt.title("Updated GP posterior mean — Function 2")
plt.legend()
plt.tight_layout()
plt.show()


# ============================================================
# 14. UCB acquisition plot
# ============================================================

acquisition_surface = acquisition.reshape(
    grid_size,
    grid_size
)

plt.figure(figsize=(10, 7))

contour = plt.contourf(
    X1_grid,
    X2_grid,
    acquisition_surface,
    levels=30
)

plt.colorbar(
    contour,
    label="UCB acquisition value"
)

plt.scatter(
    X[:, 0],
    X[:, 1],
    marker="x",
    s=90,
    label="Evaluated points"
)

plt.scatter(
    next_point[0],
    next_point[1],
    marker="*",
    s=250,
    label="Suggested next point"
)

plt.xlabel("x1")
plt.ylabel("x2")
plt.title("Updated UCB acquisition — Function 2")
plt.legend()
plt.tight_layout()
plt.show()