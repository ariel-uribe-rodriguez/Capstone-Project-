# -*- coding: utf-8 -*-
"""
Created on Fri Jul 24 20:44:26 2026

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
# 1. Load the original Function 4 data
# ============================================================

X_initial = np.load("initial_inputs.npy")
y_initial = np.load("initial_outputs.npy").reshape(-1)

print("Original input shape:", X_initial.shape)
print("Original output shape:", y_initial.shape)


# ============================================================
# 2. Add the newest real evaluation
# ============================================================

x_new = np.array([
    [0.400748, 0.397348, 0.372397, 0.430013],
    [0.443785, 0.432844, 0.281412, 0.433497],
    [0.37517,  0.408518, 0.450623, 0.429574],
    [0.398291, 0.395896, 0.399515, 0.445166],
    [0.402919, 0.439507, 0.405829, 0.410557],
    [0.368798, 0.440345, 0.396834, 0.389379],
    [0.400044, 0.433063, 0.383800, 0.423085],
    [0.402919, 0.439507, 0.405829, 0.410557],
    [0.363499, 0.459855, 0.391063, 0.436642],
    [0.401271, 0.394373, 0.386196, 0.412671],
    [0.411713, 0.401564, 0.433734, 0.387072],
    [0.417811, 0.386712, 0.428032, 0.421733]
])

y_new = np.array([
    0.383967328639116,
    -1.085852676823430,
    0.040439048471143,
    -0.150203064448482,
    0.342323011601000,
    0.107846987067759,
    0.422090379140791,
    0.342323011601000,
    -0.389663493268759,
    0.253297874505996,
    0.369258227155413,
    0.591293412728422
])

X = np.vstack([X_initial, x_new])
y = np.concatenate([y_initial, y_new])

print("Updated input shape:", X.shape)
print("Updated output shape:", y.shape)


# ============================================================
# 3. Standardise the objective values
# ============================================================

y_scaler = StandardScaler()

y_scaled = y_scaler.fit_transform(
    y.reshape(-1, 1)
).ravel()


# ============================================================
# 4. Define and fit the Gaussian Process surrogate
# ============================================================

kernel = (
    ConstantKernel(
        constant_value=1.0,
        constant_value_bounds=(1e-3, 1e3)
    )
    *
    Matern(
        length_scale=[0.2, 0.2, 0.2, 0.2],
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
    n_restarts_optimizer=10,
    random_state=4
)

gp.fit(X, y_scaled)

print("\nOptimised GP kernel:")
print(gp.kernel_)


# ============================================================
# 5. Generate candidate configurations
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
# 7. UCB acquisition function
# ============================================================

# UCB(x) = mean(x) + kappa * standard deviation(x)

kappa = 1.96

acquisition = (
    posterior_mean_scaled
    + kappa * posterior_std_scaled
)


# ============================================================
# 8. Avoid resampling existing configurations
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
# 9. Select the next configuration
# ============================================================

next_index = np.argmax(acquisition)

next_point = X_candidates[next_index]
next_predicted_mean = posterior_mean[next_index]
next_predicted_std = posterior_std[next_index]


# ============================================================
# 10. Identify the best observed configuration
# ============================================================

best_observed_index = np.argmax(y)

best_observed_point = X[best_observed_index]
best_observed_value = y[best_observed_index]


# ============================================================
# 11. Identify the maximum GP posterior mean
# ============================================================

best_mean_index = np.argmax(posterior_mean)

best_surrogate_point = X_candidates[best_mean_index]
best_surrogate_mean = posterior_mean[best_mean_index]
best_surrogate_std = posterior_std[best_mean_index]


# ============================================================
# 12. Print the results
# ============================================================

print("\n" + "=" * 74)
print("NEW REAL EVALUATION")
print("=" * 74)

print(
    f"x1 = {x_new[0, 0]:.6f}, "
    f"x2 = {x_new[0, 1]:.6f}, "
    f"x3 = {x_new[0, 2]:.6f}, "
    f"x4 = {x_new[0, 3]:.6f}"
)

print(
    f"Real objective value = "
    f"{y_new[0]:.12f}"
)


print("\n" + "=" * 74)
print("BEST OBSERVED CONFIGURATION SO FAR")
print("=" * 74)

print(
    f"x1 = {best_observed_point[0]:.6f}, "
    f"x2 = {best_observed_point[1]:.6f}, "
    f"x3 = {best_observed_point[2]:.6f}, "
    f"x4 = {best_observed_point[3]:.6f}"
)

print(
    f"Observed objective = "
    f"{best_observed_value:.12f}"
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
    f"Predicted mean = "
    f"{best_surrogate_mean:.12f}"
)

print(
    f"Predicted standard deviation = "
    f"{best_surrogate_std:.12f}"
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
    f"Predicted mean = "
    f"{next_predicted_mean:.12f}"
)

print(
    f"Predicted standard deviation = "
    f"{next_predicted_std:.12f}"
)

print(f"UCB exploration parameter = {kappa}")


# ============================================================
# 13. Plot observed versus fitted values
# ============================================================

training_mean_scaled, _ = gp.predict(
    X,
    return_std=True
)

training_predictions = y_scaler.inverse_transform(
    training_mean_scaled.reshape(-1, 1)
).ravel()

plt.figure(figsize=(8, 6))

plt.scatter(
    y,
    training_predictions,
    s=85,
    label="Evaluated configurations"
)

plot_min = min(y.min(), training_predictions.min())
plot_max = max(y.max(), training_predictions.max())

plt.plot(
    [plot_min, plot_max],
    [plot_min, plot_max],
    linestyle="--",
    label="Perfect prediction"
)

plt.xlabel("Observed objective")
plt.ylabel("GP-predicted objective")
plt.title("Observed versus predicted values — Function 4")
plt.legend()
plt.tight_layout()
plt.show()


# ============================================================
# 14. Pairwise posterior projection
# ============================================================

# Vary x1 and x2 while holding x3 and x4 at the newest
# high-performing configuration.

grid_size = 120

x1_values = np.linspace(0, 1, grid_size)
x2_values = np.linspace(0, 1, grid_size)

X1_grid, X2_grid = np.meshgrid(
    x1_values,
    x2_values
)

projection_candidates = np.column_stack(
    [
        X1_grid.ravel(),
        X2_grid.ravel(),
        np.full(X1_grid.size, x_new[0, 2]),
        np.full(X1_grid.size, x_new[0, 3])
    ]
)

projection_mean_scaled = gp.predict(
    projection_candidates
)

projection_mean = y_scaler.inverse_transform(
    projection_mean_scaled.reshape(-1, 1)
).reshape(grid_size, grid_size)

plt.figure(figsize=(9, 7))

contour = plt.contourf(
    X1_grid,
    X2_grid,
    projection_mean,
    levels=30
)

plt.colorbar(
    contour,
    label="Predicted objective"
)

plt.scatter(
    X_initial[:, 0],
    X_initial[:, 1],
    marker="x",
    s=75,
    label="Initial configurations"
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
    s=240,
    label="Suggested next configuration"
)

plt.xlabel("x1")
plt.ylabel("x2")

plt.title(
    "Updated GP posterior projection — Function 4\n"
    f"x3 = {x_new[0, 2]:.3f}, "
    f"x4 = {x_new[0, 3]:.3f}"
)

plt.legend()
plt.tight_layout()
plt.show()