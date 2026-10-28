# -*- coding: utf-8 -*-
"""
Created on Fri Jul 24 18:07:20 2026

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
# 1. Load the original Function 3 data
# ============================================================

X_initial = np.load("initial_inputs.npy")
y_initial = np.load("initial_outputs.npy").reshape(-1)

print("Original input shape:", X_initial.shape)
print("Original output shape:", y_initial.shape)


# ============================================================
# 2. Add the newest real experiment
# ============================================================

x_new = np.array([[0.364325, 0.025035, 0.000040],
                  [0.013729, 0.003267, 0.650994],
                  [0.970735, 0.997093, 0.855029],
                  [0.996491, 0.945514, 0.398090],
                  [0.986390, 0.997230, 0.508508],
                  [0.993770, 0.016602, 0.096760],
                  [0.876179, 0.000571, 0.857786],
                  [0.544863, 0.998174, 0.068425],
                  [0.928051, 0.001545, 0.530355],
                  [0.384602, 0.048026, 0.509259],
                  [0.993833, 0.390341, 0.401887],
                  [0.727271, 0.214684, 0.000098]
                  ])

y_new = np.array([-0.184054200087251,
                  -0.160908633242746,
                  -0.067269641842912,
                  -0.034269228846406,
                  -0.047122152580112,
                  -0.119886722674482,
                  -0.109736941561941,
                  -0.097491248242134,
                  -0.079650273326996,
                  -0.095478540168930,
                  -0.036209720035477,
                  -0.139572395616096
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
# 4. Define and fit the Gaussian Process
# ============================================================

kernel = (
    ConstantKernel(
        constant_value=1.0,
        constant_value_bounds=(1e-3, 1e3)
    )
    *
    Matern(
        length_scale=[0.2, 0.2, 0.2],
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
# 5. Generate candidate compound combinations
# ============================================================

rng = np.random.default_rng(4)

number_of_candidates = 200_000

X_candidates = rng.uniform(
    low=np.zeros(3),
    high=np.ones(3),
    size=(number_of_candidates, 3)
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

kappa = 1.96

acquisition = (
    posterior_mean_scaled
    + kappa * posterior_std_scaled
)


# ============================================================
# 8. Prevent resampling near existing experiments
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
# 9. Select the next experiment
# ============================================================

next_index = np.argmax(acquisition)

next_point = X_candidates[next_index]
next_predicted_mean = posterior_mean[next_index]
next_predicted_std = posterior_std[next_index]


# ============================================================
# 10. Find the best observed experiment
# ============================================================

best_observed_index = np.argmax(y)

best_observed_point = X[best_observed_index]
best_observed_value = y[best_observed_index]


# ============================================================
# 11. Find the highest GP posterior mean
# ============================================================

best_mean_index = np.argmax(posterior_mean)

best_surrogate_point = X_candidates[best_mean_index]
best_surrogate_mean = posterior_mean[best_mean_index]
best_surrogate_std = posterior_std[best_mean_index]


# ============================================================
# 12. Print the results
# ============================================================

print("\n" + "=" * 70)
print("NEW REAL EXPERIMENT")
print("=" * 70)

print(
    f"x1 = {x_new[0, 0]:.6f}, "
    f"x2 = {x_new[0, 1]:.6f}, "
    f"x3 = {x_new[0, 2]:.6f}"
)

print(
    f"Real objective value = "
    f"{y_new[0]:.12f}"
)


print("\n" + "=" * 70)
print("BEST OBSERVED EXPERIMENT SO FAR")
print("=" * 70)

print(
    f"x1 = {best_observed_point[0]:.6f}, "
    f"x2 = {best_observed_point[1]:.6f}, "
    f"x3 = {best_observed_point[2]:.6f}"
)

print(
    f"Observed objective = "
    f"{best_observed_value:.12f}"
)


print("\n" + "=" * 70)
print("MAXIMUM GP POSTERIOR MEAN")
print("=" * 70)

print(
    f"x1 = {best_surrogate_point[0]:.6f}, "
    f"x2 = {best_surrogate_point[1]:.6f}, "
    f"x3 = {best_surrogate_point[2]:.6f}"
)

print(
    f"Predicted mean = "
    f"{best_surrogate_mean:.12f}"
)

print(
    f"Predicted standard deviation = "
    f"{best_surrogate_std:.12f}"
)


print("\n" + "=" * 70)
print("SUGGESTED NEXT SAMPLING POINT USING UCB")
print("=" * 70)

print(
    f"x1 = {next_point[0]:.6f}, "
    f"x2 = {next_point[1]:.6f}, "
    f"x3 = {next_point[2]:.6f}"
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
# 13. Visualise the evaluated and suggested experiments
# ============================================================

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection="3d")

scatter = ax.scatter(
    X_initial[:, 0],
    X_initial[:, 1],
    X_initial[:, 2],
    c=y_initial,
    s=80,
    label="Initial experiments"
)

ax.scatter(
    x_new[0, 0],
    x_new[0, 1],
    x_new[0, 2],
    marker="o",
    s=160,
    facecolors="none",
    linewidths=2,
    label="Newest real experiment"
)

ax.scatter(
    next_point[0],
    next_point[1],
    next_point[2],
    marker="*",
    s=260,
    label="Suggested next experiment"
)

fig.colorbar(
    scatter,
    ax=ax,
    label="Observed transformed objective"
)

ax.set_xlabel("Compound 1")
ax.set_ylabel("Compound 2")
ax.set_zlabel("Compound 3")
ax.set_title("Updated Bayesian optimisation — Function 3")
ax.legend()

plt.tight_layout()
plt.show()