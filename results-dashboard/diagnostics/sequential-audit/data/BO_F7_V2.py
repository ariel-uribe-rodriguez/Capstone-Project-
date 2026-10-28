# -*- coding: utf-8 -*-
"""
Created on Fri Jul 24 21:09:47 2026

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
# 1. Load the original Function 7 data
# ============================================================

X_initial = np.load("initial_inputs.npy")
y_initial = np.load("initial_outputs.npy").reshape(-1)

print("Original input shape:", X_initial.shape)
print("Original output shape:", y_initial.shape)

if X_initial.ndim != 2 or X_initial.shape[1] != 6:
    raise ValueError(
        "Function 7 inputs must have shape (n_samples, 6)."
    )

if X_initial.shape[0] != y_initial.shape[0]:
    raise ValueError(
        "The numbers of input and output observations must match."
    )


# ============================================================
# 2. Add the newest real evaluation
# ============================================================

x_new = np.array([
    [0.003850, 0.349360, 0.786090, 0.173280, 0.343800, 0.761930],
    [0.022392, 0.191815, 0.462369, 0.041344, 0.338251, 0.941049],
    [0.031752, 0.176544, 0.584333, 0.028749, 0.309210, 0.750607],
    [0.025543, 0.002700, 0.903163, 0.130992, 0.374768, 0.802957],
    [0.032469, 0.354757, 0.447308, 0.051547, 0.287663, 0.684726],
    [0.095675, 0.314063, 0.346289, 0.041216, 0.306305, 0.768184],
    [0.040698, 0.212710, 0.759622, 0.075716, 0.335790, 0.622712],
    [0.031658, 0.285704, 0.082322, 0.204186, 0.309998, 0.702371],
    [0.007329, 0.233367, 0.218872, 0.251825, 0.286766, 0.650986],
    [0.025896, 0.144693, 0.334256, 0.306807, 0.262809, 0.728651],
    [0.033059, 0.063528, 0.878726, 0.310122, 0.231504, 0.621280],
    [0.026044, 0.047348, 0.244561, 0.290118, 0.285310, 0.707193]
])

y_new = np.array([
    1.7356111016513,
    0.9688914614921,
    1.6881290679496,
    1.0163095577332,
    1.6615991266137,
    1.4956588978796,
    1.5862155116567,
    2.0220598407021,
    2.4350544508639,
    2.6067398057341,
    1.5487545831540,
    2.5126374551999
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
# 4. Define and fit the Gaussian Process
# ============================================================

# A Matérn kernel is appropriate for a nonlinear and potentially
# irregular hyperparameter-response surface.
#
# One length scale is assigned to each hyperparameter.

kernel = (
    ConstantKernel(
        constant_value=1.0,
        constant_value_bounds=(1e-3, 1e3)
    )
    *
    Matern(
        length_scale=[0.2] * 6,
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
# 5. Generate candidate hyperparameter configurations
# ============================================================

rng = np.random.default_rng(4)

number_of_candidates = 300_000

lower_bounds = np.zeros(6)
upper_bounds = np.ones(6)

X_candidates = rng.uniform(
    low=lower_bounds,
    high=upper_bounds,
    size=(number_of_candidates, 6)
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

# Maximisation:
#
# UCB(x) = mean(x) + kappa * standard deviation(x)

kappa = 1.96

acquisition = (
    posterior_mean_scaled
    + kappa * posterior_std_scaled
)


# ============================================================
# 8. Avoid sampling too close to previous configurations
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
# 9. Select the next sampling point
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

print("\n" + "=" * 78)
print("NEW REAL HYPERPARAMETER EVALUATION")
print("=" * 78)

print(
    "Hyperparameters:",
    np.round(x_new[0], 6)
)

print(
    f"Real objective value = "
    f"{y_new[0]:.12f}"
)


print("\n" + "=" * 78)
print("BEST OBSERVED CONFIGURATION SO FAR")
print("=" * 78)

print(
    "Hyperparameters:",
    np.round(best_observed_point, 6)
)

print(
    f"Observed objective = "
    f"{best_observed_value:.12f}"
)


print("\n" + "=" * 78)
print("MAXIMUM GP POSTERIOR MEAN")
print("=" * 78)

print(
    "Hyperparameters:",
    np.round(best_surrogate_point, 6)
)

print(
    f"Predicted mean = "
    f"{best_surrogate_mean:.12f}"
)

print(
    f"Predicted standard deviation = "
    f"{best_surrogate_std:.12f}"
)


print("\n" + "=" * 78)
print("SUGGESTED NEXT SAMPLING POINT USING UCB")
print("=" * 78)

print(
    "Hyperparameters:",
    np.round(next_point, 6)
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
# 13. Compare observed and GP-fitted scores
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

plt.xlabel("Observed performance")
plt.ylabel("GP-predicted performance")
plt.title("Observed versus predicted values — Function 7")
plt.legend()
plt.tight_layout()
plt.show()


# ============================================================
# 14. Plot each hyperparameter against the objective
# ============================================================

for j in range(6):

    plt.figure(figsize=(8, 5))

    plt.scatter(
        X[:, j],
        y,
        s=75,
        label="Evaluated configurations"
    )

    plt.scatter(
        next_point[j],
        next_predicted_mean,
        marker="*",
        s=230,
        label="Suggested next configuration"
    )

    plt.xlabel(f"Hyperparameter {j + 1}")
    plt.ylabel("Performance score")

    plt.title(
        f"Performance versus hyperparameter {j + 1}"
    )

    plt.legend()
    plt.tight_layout()
    plt.show()


# ============================================================
# 15. Pairwise posterior projection
# ============================================================

# Vary x3 and x6 because the current high-performance region
# appears particularly relevant along these dimensions.
# Hold the remaining variables at the posterior-mean optimum.

grid_size = 120

x3_values = np.linspace(0, 1, grid_size)
x6_values = np.linspace(0, 1, grid_size)

X3_grid, X6_grid = np.meshgrid(
    x3_values,
    x6_values
)

projection_candidates = np.tile(
    best_surrogate_point,
    (X3_grid.size, 1)
)

projection_candidates[:, 2] = X3_grid.ravel()
projection_candidates[:, 5] = X6_grid.ravel()

projection_mean_scaled = gp.predict(
    projection_candidates
)

projection_mean = y_scaler.inverse_transform(
    projection_mean_scaled.reshape(-1, 1)
).reshape(grid_size, grid_size)

plt.figure(figsize=(9, 7))

contour = plt.contourf(
    X3_grid,
    X6_grid,
    projection_mean,
    levels=30
)

plt.colorbar(
    contour,
    label="Predicted performance"
)

plt.scatter(
    X_initial[:, 2],
    X_initial[:, 5],
    marker="x",
    s=75,
    label="Initial configurations"
)

plt.scatter(
    x_new[0, 2],
    x_new[0, 5],
    marker="o",
    s=150,
    facecolors="none",
    linewidths=2,
    label="Newest real evaluation"
)

plt.scatter(
    best_surrogate_point[2],
    best_surrogate_point[5],
    marker="D",
    s=120,
    label="Maximum posterior mean"
)

plt.scatter(
    next_point[2],
    next_point[5],
    marker="*",
    s=240,
    label="Suggested next configuration"
)

plt.xlabel("x3")
plt.ylabel("x6")

plt.title(
    "Updated GP posterior projection — Function 7\n"
    "Other hyperparameters fixed at posterior-mean optimum"
)

plt.legend()
plt.tight_layout()
plt.show()