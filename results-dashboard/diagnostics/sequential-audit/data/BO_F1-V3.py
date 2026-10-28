# -*- coding: utf-8 -*-
"""
Created on Fri Jul 24 17:37:09 2026

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
# 1. Load the original observations
# ============================================================

X_initial = np.load("initial_inputs.npy")
y_initial = np.load("initial_outputs.npy").reshape(-1)

print("Original input shape:", X_initial.shape)
print("Original output shape:", y_initial.shape)


# ============================================================
# 2. Add the newly evaluated point
# ============================================================

# Week 1 result supplied from the real unknown function
x_new = np.array([[0.705, 0.000],
                  [0.373333, 0.153333],
                  [0.436667, 0.333333],
                  [0.753333, 0.523333],
                  [1.000000, 0.463333],
                  [1.000000, 0.953333],
                  [0.800000, 0.500000],
                  [0.536667, 0.120000],
                  [0.493333, 0.236667],
                  [0.960000, 0.703333],
                  [0.196667, 1.000000],
                  [0.920000, 0.640000]
                  ])
y_new = np.array([-7.04764982228395e-181, 
                  2.52224567185068E-53,
                  -2.10894622516894E-06,
                  -8.18177814707581E-20,
                  -1.78614939837502E-115,
                  -7.90106517685532E-170,
                  8.60679902398953E-33,
                  -3.86090875092912E-74,
                  -2.06706029014074E-28,
                  -8.97136097295711E-81,
                  1.66888744815094E-228,
                  8.37394249550469E-60
                  ])

X = np.vstack([X_initial, x_new])
y = np.concatenate([y_initial, y_new])

print("Updated input shape:", X.shape)
print("Updated output shape:", y.shape)


# ============================================================
# 3. Standardise the objective values
# ============================================================

epsilon = 1e-8

y_transformed = (
    np.sign(y)
    * np.log1p(np.abs(y) / epsilon)
    / np.log(10.0)
)

y_scaler = StandardScaler()

y_scaled = y_scaler.fit_transform(
    y_transformed.reshape(-1, 1)
).ravel()


# ============================================================
# 4. Define and fit the Gaussian Process surrogate
# ============================================================

# Matérn is useful here because the unknown function may contain
# narrow, localised peaks rather than being globally very smooth.

kernel = (
    ConstantKernel(
        constant_value=1.0,
        constant_value_bounds=(1e-3, 1e3)
    )
    *
    Matern(
        length_scale=[0.2, 0.2],
        length_scale_bounds=(1e-3, 50.0),
        nu=2.5
    )
    +
    WhiteKernel(
        noise_level=1e-8,
        noise_level_bounds=(1e-12, 1e-2)
    )
)

gp = GaussianProcessRegressor(
    kernel=kernel,
    alpha=1e-10,
    normalize_y=False,
    n_restarts_optimizer=10,
    random_state=4
)

gp.fit(X, y_scaled)

print("\nOptimised GP kernel:")
print(gp.kernel_)


# ============================================================
# 5. Generate a dense 2D candidate grid
# ============================================================

grid_size = 301

x1_values = np.linspace(0, 1, grid_size)
x2_values = np.linspace(0, 1, grid_size)

X1_grid, X2_grid = np.meshgrid(
    x1_values,
    x2_values
)

X_candidates = np.column_stack(
    [
        X1_grid.ravel(),
        X2_grid.ravel()
    ]
)


# ============================================================
# 6. Predict posterior mean and uncertainty
# ============================================================

posterior_mean_scaled, posterior_std_scaled = gp.predict(
    X_candidates,
    return_std=True
)


# Back to signed-log scale
posterior_mean_transformed = y_scaler.inverse_transform(
    posterior_mean_scaled.reshape(-1, 1)
).ravel()

# Back to original objective scale
posterior_mean = (
    np.sign(posterior_mean_transformed)
    * epsilon
    * (10.0 ** np.abs(posterior_mean_transformed) - 1.0)
)

posterior_std = (
    posterior_std_scaled
    * y_scaler.scale_[0]
)

posterior_std_transformed = (
    posterior_std_scaled * y_scaler.scale_[0]
)

lower_transformed = (
    posterior_mean_transformed
    - 1.96 * posterior_std_transformed
)

upper_transformed = (
    posterior_mean_transformed
    + 1.96 * posterior_std_transformed
)

def inverse_signed_log(z, epsilon):
    return (
        np.sign(z)
        * epsilon
        * (10.0 ** np.abs(z) - 1.0)
    )

posterior_lower = inverse_signed_log(
    lower_transformed, epsilon
)

posterior_upper = inverse_signed_log(
    upper_transformed, epsilon
)

# ============================================================
# 7. UCB acquisition function
# ============================================================

# Maximisation:
# UCB(x) = mean(x) + kappa * standard deviation(x)

kappa = 0.0 +0.0*1.96

acquisition = (
    posterior_mean_scaled
    + kappa * posterior_std_scaled
)


# ============================================================
# 8. Prevent resampling points too close to previous observations
# ============================================================

minimum_distance = 0.01

distances = np.sqrt(
    (
        (
            X_candidates[:, None, :]
            - X[None, :, :]
        ) ** 2
    ).sum(axis=2)
)

closest_distance = distances.min(axis=1)

acquisition[
    closest_distance < minimum_distance
] = -np.inf


# ============================================================
# 8A. Maximum posterior uncertainty candidate
# ============================================================

uncertainty_acquisition = posterior_std_scaled.copy()

# Apply the same minimum-distance restriction used for UCB
uncertainty_acquisition[
    closest_distance < minimum_distance
] = -np.inf

max_uncertainty_index = np.argmax(
    uncertainty_acquisition
)

max_uncertainty_point = X_candidates[
    max_uncertainty_index
]

max_uncertainty_mean = posterior_mean[
    max_uncertainty_index
]

max_uncertainty_std = posterior_std_scaled[max_uncertainty_index]

# ============================================================
# 9. Select the next suggested point
# ============================================================

next_index = np.argmax(acquisition)

next_point = X_candidates[next_index]
next_predicted_mean = posterior_mean[next_index]
next_predicted_std = posterior_std[next_index]


# ============================================================
# 10. Identify current best observed point
# ============================================================

best_observed_index = np.argmax(y)

best_observed_point = X[best_observed_index]
best_observed_value = y[best_observed_index]


# ============================================================
# 11. Identify maximum GP posterior mean
# ============================================================

best_mean_index = np.argmax(posterior_mean)

best_surrogate_point = X_candidates[best_mean_index]
best_surrogate_mean = posterior_mean[best_mean_index]
best_surrogate_std = posterior_std[best_mean_index]


# ============================================================
# 12. Print results
# ============================================================

print("\n" + "=" * 66)
print("NEW OBSERVATION ADDED")
print("=" * 66)

print(
    f"Evaluated point: "
    f"x1 = {x_new[0, 0]:.6f}, "
    f"x2 = {x_new[0, 1]:.6f}"
)

print(
    f"Real objective value: "
    f"{y_new[0]:.12e}"
)


print("\n" + "=" * 66)
print("BEST OBSERVED POINT SO FAR")
print("=" * 66)

print(
    f"x1 = {best_observed_point[0]:.6f}, "
    f"x2 = {best_observed_point[1]:.6f}"
)

print(
    f"Observed objective: "
    f"{best_observed_value:.12e}"
)


print("\n" + "=" * 66)
print("MAXIMUM ACCORDING TO GP POSTERIOR MEAN")
print("=" * 66)

print(
    f"x1 = {best_surrogate_point[0]:.6f}, "
    f"x2 = {best_surrogate_point[1]:.6f}"
)

print(
    f"Predicted mean: "
    f"{best_surrogate_mean:.12e}"
)

print(
    f"Predicted standard deviation: "
    f"{best_surrogate_std:.12e}"
)


print("\n" + "=" * 66)
print("SUGGESTED NEXT SAMPLING POINT USING UCB")
print("=" * 66)

print(
    f"x1 = {next_point[0]:.6f}, "
    f"x2 = {next_point[1]:.6f}"
)

print(
    f"Predicted mean: "
    f"{next_predicted_mean:.12e}"
)

print(
    f"Predicted standard deviation: "
    f"{next_predicted_std:.12e}"
)

print(f"UCB exploration parameter: {kappa}")

print('new diagnostic')

print("\n" + "=" * 66)
print("MAXIMUM POSTERIOR UNCERTAINTY POINT")
print("=" * 66)

print(
    f"x1 = {max_uncertainty_point[0]:.6f}, "
    f"x2 = {max_uncertainty_point[1]:.6f}"
)

print(
    f"Predicted mean: "
    f"{max_uncertainty_mean:.12e}"
)

print(
    f"Predicted standard deviation: "
    f"{max_uncertainty_std:.12e}"
)

print(f"Standard deviation (standardized signed-log scale): {max_uncertainty_std:.12e}")


# ============================================================
# 13. Plot posterior mean
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
    s=140,
    facecolors="none",
    linewidths=2,
    label="Newest real evaluation"
)

plt.scatter(
    next_point[0],
    next_point[1],
    marker="*",
    s=240,
    label="Suggested next point"
)

plt.xlabel("x1")
plt.ylabel("x2")
plt.title("Updated Gaussian Process posterior mean")
plt.legend()
plt.tight_layout()
plt.show()


# ============================================================
# 14. Plot UCB acquisition surface
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
    s=240,
    label="Suggested next point"
)

plt.xlabel("x1")
plt.ylabel("x2")
plt.title("Updated UCB acquisition function")
plt.legend()
plt.tight_layout()
plt.show()