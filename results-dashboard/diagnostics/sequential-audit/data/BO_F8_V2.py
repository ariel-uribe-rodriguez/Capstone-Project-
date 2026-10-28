# -*- coding: utf-8 -*-
"""
Created on Fri Jul 24 21:30:49 2026

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
# 1. Load the original Function 8 data
# ============================================================

X_initial = np.load("initial_inputs.npy")
y_initial = np.load("initial_outputs.npy").reshape(-1)

print("Original input shape:", X_initial.shape)
print("Original output shape:", y_initial.shape)

if X_initial.ndim != 2 or X_initial.shape[1] != 8:
    raise ValueError(
        "Function 8 inputs must have shape (n_samples, 8)."
    )

if X_initial.shape[0] != y_initial.shape[0]:
    raise ValueError(
        "The numbers of input and output observations must match."
    )

if not np.all(np.isfinite(X_initial)) or not np.all(
    np.isfinite(y_initial)
):
    raise ValueError("The initial data contain invalid values.")


# ============================================================
# 2. Add the newest real evaluation
# ============================================================

x_new = np.array([
    [0.074230, 0.036380, 0.035470, 0.013620, 0.901480, 0.496720, 0.073160, 0.263460],
    [0.148809, 0.170651, 0.063751, 0.009482, 0.972712, 0.172801, 0.05619, 0.933778],
    [0.233797, 0.083312, 0.188173, 0.017602, 0.970989, 0.313072, 0.230576, 0.394151],
    [0.013846, 0.076278, 0.057927, 0.899005, 0.979142, 0.797222, 0.158265, 0.403537],
    [0.078076, 0.065295, 0.247625, 0.150762, 0.968037, 0.885292, 0.024554, 0.765675],
    [0.019104, 0.452096, 0.002271, 0.045690, 0.977445, 0.568970, 0.091003, 0.153947],
    [0.028306, 0.064779, 0.026427, 0.193804, 0.980014, 0.754845, 0.498409, 0.017575],
    [0.142636, 0.005540, 0.001637, 0.733842, 0.980240, 0.267332, 0.036697, 0.023205],
    [0.010742, 0.025282, 0.061676, 0.109365, 0.940626, 0.318616, 0.256070, 0.998736],
    [0.009130, 0.223934, 0.059284, 0.037977, 0.134663, 0.423299, 0.037659, 0.936840],
    [0.082517, 0.206942, 0.011868, 0.044798, 0.981148, 0.538385, 0.095057, 0.678106],
    [0.019898, 0.112955, 0.097346, 0.165204, 0.672542, 0.424253, 0.207721, 0.458800]
])

y_new = np.array([
    9.89169250074,
    9.78741949031,
    9.87640005117,
    9.720569275,
    9.72348067656,
    9.77166776792,
    9.65519018774,
    9.42821037729,
    9.88788100823,
    9.65919261504,
    9.90269650058,
    9.96639164383
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

# One length scale is fitted for each input variable.
# This allows the GP to represent different sensitivities
# along the eight dimensions.

kernel = (
    ConstantKernel(
        constant_value=1.0,
        constant_value_bounds=(1e-3, 1e3)
    )
    *
    Matern(
        length_scale=[0.2] * 8,
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
    n_restarts_optimizer=5,
    random_state=4
)

gp.fit(X, y_scaled)

print("\nOptimised GP kernel:")
print(gp.kernel_)


# ============================================================
# 5. Generate candidate sampling points
# ============================================================

# A full 8D grid is impractical, so a large random candidate
# set is used.

rng = np.random.default_rng(4)

number_of_candidates = 300_000

lower_bounds = np.zeros(8)
upper_bounds = np.ones(8)

X_candidates = rng.uniform(
    low=lower_bounds,
    high=upper_bounds,
    size=(number_of_candidates, 8)
)


# ============================================================
# 6. Compute posterior mean and uncertainty
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
# 7. Define the UCB acquisition function
# ============================================================

# For maximisation:
#
# UCB(x) = mean(x) + kappa * standard deviation(x)

kappa = 1.96

acquisition = (
    posterior_mean_scaled
    + kappa * posterior_std_scaled
)


# ============================================================
# 8. Prevent resampling close to existing observations
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
# 9. Select the next suggested sampling point
# ============================================================

next_index = np.argmax(acquisition)

next_point = X_candidates[next_index]
next_predicted_mean = posterior_mean[next_index]
next_predicted_std = posterior_std[next_index]


# ============================================================
# 10. Identify the best observed point
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

print("\n" + "=" * 82)
print("NEW REAL EVALUATION")
print("=" * 82)

print(
    "Parameters:",
    np.round(x_new[0], 6)
)

print(
    f"Real objective value = "
    f"{y_new[0]:.12f}"
)


print("\n" + "=" * 82)
print("BEST OBSERVED CONFIGURATION SO FAR")
print("=" * 82)

print(
    "Parameters:",
    np.round(best_observed_point, 6)
)

print(
    f"Observed objective = "
    f"{best_observed_value:.12f}"
)


print("\n" + "=" * 82)
print("MAXIMUM GP POSTERIOR MEAN")
print("=" * 82)

print(
    "Parameters:",
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


print("\n" + "=" * 82)
print("SUGGESTED NEXT SAMPLING POINT USING UCB")
print("=" * 82)

print(
    "Parameters:",
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
# 13. Compare observed and GP-fitted responses
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
plt.title("Observed versus predicted values — Function 8")
plt.legend()
plt.tight_layout()
plt.show()


# ============================================================
# 14. Plot each input variable against the objective
# ============================================================

for j in range(8):

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
        label="Suggested next point"
    )

    plt.xlabel(f"Input variable {j + 1}")
    plt.ylabel("Objective value")

    plt.title(
        f"Objective versus input variable {j + 1}"
    )

    plt.legend()
    plt.tight_layout()
    plt.show()


# ============================================================
# 15. Create a pairwise GP projection
# ============================================================

# Vary x1 and x8 while fixing the remaining variables at
# the maximum-posterior-mean configuration.

grid_size = 120

x1_values = np.linspace(0, 1, grid_size)
x8_values = np.linspace(0, 1, grid_size)

X1_grid, X8_grid = np.meshgrid(
    x1_values,
    x8_values
)

projection_candidates = np.tile(
    best_surrogate_point,
    (X1_grid.size, 1)
)

projection_candidates[:, 0] = X1_grid.ravel()
projection_candidates[:, 7] = X8_grid.ravel()

projection_mean_scaled = gp.predict(
    projection_candidates
)

projection_mean = y_scaler.inverse_transform(
    projection_mean_scaled.reshape(-1, 1)
).reshape(grid_size, grid_size)

plt.figure(figsize=(9, 7))

contour = plt.contourf(
    X1_grid,
    X8_grid,
    projection_mean,
    levels=30
)

plt.colorbar(
    contour,
    label="Predicted objective"
)

plt.scatter(
    X_initial[:, 0],
    X_initial[:, 7],
    marker="x",
    s=75,
    label="Initial configurations"
)

plt.scatter(
    x_new[0, 0],
    x_new[0, 7],
    marker="o",
    s=150,
    facecolors="none",
    linewidths=2,
    label="Newest real evaluation"
)

plt.scatter(
    best_surrogate_point[0],
    best_surrogate_point[7],
    marker="D",
    s=120,
    label="Maximum posterior mean"
)

plt.scatter(
    next_point[0],
    next_point[7],
    marker="*",
    s=240,
    label="Suggested next point"
)

plt.xlabel("x1")
plt.ylabel("x8")

plt.title(
    "Updated GP posterior projection — Function 8\n"
    "Other six variables fixed at posterior-mean optimum"
)

plt.legend()
plt.tight_layout()
plt.show()