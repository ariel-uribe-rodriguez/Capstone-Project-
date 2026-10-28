# -*- coding: utf-8 -*-
"""
Created on Fri Jul 24 20:58:48 2026

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
# 1. Load the original Function 6 data
# ============================================================

X_initial = np.load("initial_inputs.npy")
y_initial = np.load("initial_outputs.npy").reshape(-1)

print("Original input shape:", X_initial.shape)
print("Original output shape:", y_initial.shape)


# ============================================================
# 2. Add the newest real evaluation
# ============================================================

x_new = np.array([
    [0.389470, 0.097120, 0.933150, 0.996500, 0.009780],
    [0.420725, 0.218727, 0.300242, 0.930405, 0.016763],
    [0.542914, 0.391549, 0.635480, 0.745057, 0.013481],
    [0.543812, 0.417979, 0.984392, 0.687242, 0.022604],
    [0.484863, 0.470713, 0.557798, 0.703790, 0.011037],
    [0.573471, 0.405208, 0.630630, 0.970393, 0.016574],
    [0.561845, 0.091087, 0.514127, 0.607532, 0.007939],
    [0.454996, 0.446436, 0.600632, 0.666534, 0.009577],
    [0.458899, 0.331670, 0.646890, 0.780744, 0.005993],
    [0.217714, 0.351481, 0.685785, 0.838972, 0.005859],
    [0.445267, 0.391132, 0.597170, 0.766116, 0.034240],
    [0.407516, 0.331925, 0.635851, 0.807692, 0.539437]
])

y_new = np.array([
    -0.903500264044212,
    -0.749096505337708,
    -0.313816865282233,
    -0.663162613131903,
    -0.387909203108265,
    -0.570058821370732,
    -0.861885593910301,
    -0.376711239004262,
    -0.111470908299446,
    -0.463318685362006,
    -0.130017491144102,
    -0.422510384508492
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
# 4. Define and fit the Gaussian Process surrogate
# ============================================================

kernel = (
    ConstantKernel(
        constant_value=1.0,
        constant_value_bounds=(1e-3, 1e3)
    )
    *
    Matern(
        length_scale=[0.2, 0.2, 0.2, 0.2, 0.2],
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
# 5. Generate candidate recipes
# ============================================================

rng = np.random.default_rng(4)

number_of_candidates = 250_000

lower_bounds = np.zeros(5)
upper_bounds = np.ones(5)

X_candidates = rng.uniform(
    low=lower_bounds,
    high=upper_bounds,
    size=(number_of_candidates, 5)
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

# For maximisation:
# UCB(x) = mean(x) + kappa * standard deviation(x)

kappa = 1.96

acquisition = (
    posterior_mean_scaled
    + kappa * posterior_std_scaled
)


# ============================================================
# 8. Avoid sampling too close to existing recipes
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
# 9. Select the next suggested recipe
# ============================================================

next_index = np.argmax(acquisition)

next_point = X_candidates[next_index]
next_predicted_mean = posterior_mean[next_index]
next_predicted_std = posterior_std[next_index]


# ============================================================
# 10. Identify the best observed recipe
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
# 12. Print results
# ============================================================

print("\n" + "=" * 76)
print("NEW REAL RECIPE EVALUATION")
print("=" * 76)

print(
    f"x1 = {x_new[0, 0]:.6f}, "
    f"x2 = {x_new[0, 1]:.6f}, "
    f"x3 = {x_new[0, 2]:.6f}, "
    f"x4 = {x_new[0, 3]:.6f}, "
    f"x5 = {x_new[0, 4]:.6f}"
)

print(
    f"Real objective value = "
    f"{y_new[0]:.12f}"
)


print("\n" + "=" * 76)
print("BEST OBSERVED RECIPE SO FAR")
print("=" * 76)

print(
    f"x1 = {best_observed_point[0]:.6f}, "
    f"x2 = {best_observed_point[1]:.6f}, "
    f"x3 = {best_observed_point[2]:.6f}, "
    f"x4 = {best_observed_point[3]:.6f}, "
    f"x5 = {best_observed_point[4]:.6f}"
)

print(
    f"Observed objective = "
    f"{best_observed_value:.12f}"
)


print("\n" + "=" * 76)
print("MAXIMUM GP POSTERIOR MEAN")
print("=" * 76)

print(
    f"x1 = {best_surrogate_point[0]:.6f}, "
    f"x2 = {best_surrogate_point[1]:.6f}, "
    f"x3 = {best_surrogate_point[2]:.6f}, "
    f"x4 = {best_surrogate_point[3]:.6f}, "
    f"x5 = {best_surrogate_point[4]:.6f}"
)

print(
    f"Predicted mean = "
    f"{best_surrogate_mean:.12f}"
)

print(
    f"Predicted standard deviation = "
    f"{best_surrogate_std:.12f}"
)


print("\n" + "=" * 76)
print("SUGGESTED NEXT SAMPLING POINT USING UCB")
print("=" * 76)

print(
    f"x1 = {next_point[0]:.6f}, "
    f"x2 = {next_point[1]:.6f}, "
    f"x3 = {next_point[2]:.6f}, "
    f"x4 = {next_point[3]:.6f}, "
    f"x5 = {next_point[4]:.6f}"
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
# 13. Compare observed and fitted scores
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
    label="Evaluated recipes"
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
plt.title("Observed versus predicted recipe scores — Function 6")
plt.legend()
plt.tight_layout()
plt.show()


# ============================================================
# 14. Plot each ingredient against the objective
# ============================================================

ingredient_names = [
    "Flour",
    "Sugar",
    "Eggs",
    "Butter",
    "Milk"
]

for j, ingredient_name in enumerate(ingredient_names):

    plt.figure(figsize=(8, 5))

    plt.scatter(
        X[:, j],
        y,
        s=75,
        label="Evaluated recipes"
    )

    plt.scatter(
        next_point[j],
        next_predicted_mean,
        marker="*",
        s=230,
        label="Suggested next recipe"
    )

    plt.xlabel(ingredient_name)
    plt.ylabel("Objective value")
    plt.title(f"Objective versus {ingredient_name}")
    plt.legend()
    plt.tight_layout()
    plt.show()


# ============================================================
# 15. Pairwise posterior projection
# ============================================================

# Vary x1 and x4 while fixing the remaining variables at the
# maximum-posterior-mean recipe.

grid_size = 120

x1_values = np.linspace(0, 1, grid_size)
x4_values = np.linspace(0, 1, grid_size)

X1_grid, X4_grid = np.meshgrid(
    x1_values,
    x4_values
)

projection_candidates = np.tile(
    best_surrogate_point,
    (X1_grid.size, 1)
)

projection_candidates[:, 0] = X1_grid.ravel()
projection_candidates[:, 3] = X4_grid.ravel()

projection_mean_scaled = gp.predict(
    projection_candidates
)

projection_mean = y_scaler.inverse_transform(
    projection_mean_scaled.reshape(-1, 1)
).reshape(grid_size, grid_size)

plt.figure(figsize=(9, 7))

contour = plt.contourf(
    X1_grid,
    X4_grid,
    projection_mean,
    levels=30
)

plt.colorbar(
    contour,
    label="Predicted objective"
)

plt.scatter(
    X_initial[:, 0],
    X_initial[:, 3],
    marker="x",
    s=75,
    label="Initial recipes"
)

plt.scatter(
    x_new[0, 0],
    x_new[0, 3],
    marker="o",
    s=150,
    facecolors="none",
    linewidths=2,
    label="Newest real evaluation"
)

plt.scatter(
    next_point[0],
    next_point[3],
    marker="*",
    s=240,
    label="Suggested next recipe"
)

plt.xlabel("x1")
plt.ylabel("x4")

plt.title(
    "Updated GP posterior projection — Function 6\n"
    "Other ingredients fixed at surrogate-best values"
)

plt.legend()
plt.tight_layout()
plt.show()