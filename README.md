# Black-Box Optimisation (BBO) Capstone

## Project Overview

### Description

This project investigates Black-Box Optimisation (BBO) techniques for solving
optimisation problems where the mathematical form of the objective function is
unknown.

Instead of directly analysing the objective function, the optimisation framework
sequentially evaluates candidate solutions, builds a surrogate model from the
collected observations, and proposes new sampling locations expected to improve
the objective.

The project considers eight benchmark optimisation problems with dimensionality
ranging from 2 to 8 decision variables. Each function represents an expensive
black-box system for which only input-output observations are available.

### Project Goal

The primary objective is to maximise an unknown objective function while using
as few function evaluations as possible.

Since each evaluation is assumed to be expensive, the optimisation strategy
must balance:

- exploration of uncertain or poorly sampled regions;
- exploitation of regions with promising predicted objective values;
- progressive improvement of the surrogate model; and
- efficient use of the limited evaluation budget.

This setting resembles many real-world engineering and machine-learning
applications, including experimental design, simulation-based optimisation,
hyperparameter optimisation and industrial process optimisation.

---

## Inputs and Outputs

### Inputs

Each black-box function receives a vector of continuous decision variables:

- the number of variables depends on the function;
- dimensionality ranges from 2 to 8;
- each variable is bounded between 0 and 1; and
- one candidate point is submitted during each optimisation round.

Example:

`x = [0.62, 0.14, 0.91, 0.35]`

### Outputs

The black-box function returns:

- a single objective value;
- no analytical derivatives; and
- no information about the underlying mathematical structure.

After each evaluation, the optimisation framework updates the surrogate model
and calculates:

- GP posterior mean;
- predictive uncertainty;
- acquisition-function values; and
- the next suggested sampling point.

---

## Optimisation Approach

### Gaussian Process Surrogate

The unknown objective functions are approximated using Gaussian Process (GP)
regression with a Matérn kernel.

After each new observation, the GP is refitted using all observations available
at that optimisation round. The surrogate provides both an estimate of the
objective function and an estimate of predictive uncertainty.

The Matérn kernel was selected because it provides flexibility for representing
smooth as well as moderately irregular response surfaces.

### Upper Confidence Bound Acquisition

Candidate points are selected using the Upper Confidence Bound (UCB)
acquisition function:

**UCB(x) = μ(x) + κσ(x)**

where:

- **μ(x)** is the GP posterior mean;
- **σ(x)** is the posterior standard deviation; and
- **κ** controls the exploration-exploitation trade-off.

The working implementation uses **κ = 1.96**.

Previously sampled or very close points are excluded using a minimum-distance
criterion to reduce duplicate evaluations.

---

## Sequential Optimisation Strategy

The optimisation follows an iterative procedure:

1. Add the newly evaluated input-output observation.
2. Update the Gaussian Process surrogate.
3. Recompute posterior mean and uncertainty.
4. Evaluate the UCB acquisition function.
5. Select the next sampling point.
6. Evaluate the black-box function.
7. Repeat until the evaluation budget is exhausted.

The same general GP-UCB framework was applied across the eight functions.
However, diagnostic analysis was used when the observed behaviour indicated
that a function required additional treatment rather than modifying parameters
arbitrarily.

### Function 1 Diagnostic Refinement

Function 1 exhibited an extreme objective-value distribution. Most observations
were numerically very close to zero, while one initial observation was
approximately `-3.606e-3`.

Diagnostic analysis showed that percentage prediction error was inappropriate
for this function because the denominator was frequently close to zero.

For the tenth-round model, a signed-log transformation was therefore introduced:

**z(y) = sign(y) log10(1 + |y| / ε)**

with:

**ε = 1e-8**

The transformation compresses the extreme dynamic range while preserving the
sign and ordering of the observations.

This modification was specific to Function 1; it was not applied automatically
to the remaining functions.

---

## Evaluation

The optimisation process is evaluated using several complementary indicators:

- best objective value observed so far;
- progression of the objective across optimisation rounds;
- GP posterior mean;
- posterior uncertainty;
- prediction residuals; and
- predictive-interval coverage where appropriate.

Prediction accuracy alone is not treated as the optimisation objective. A
surrogate can provide useful optimisation guidance even when individual point
predictions contain error.

The results should therefore be interpreted as the best solutions identified
under a limited sequential-query budget rather than proof that the global
optimum has been located.

---

## Repository Structure

The repository is being developed progressively. The intended structure is:

    Capstone-Project-/
    │
    ├── README.md
    │
    ├── docs/
    │   ├── BBO_Capstone_Datasheet.xlsx
    │   └── BBO_Model_Card_v2.docx
    │
    ├── data/
    │   ├── initial/
    │   └── observations/
    │
    ├── src/
    │
    ├── notebooks/
    │
    └── figures/

Additional datasets, optimisation scripts, notebooks and figures will be added
as the project repository is completed.

---

## Documentation

Detailed information about the dataset and optimisation methodology is available
in the `docs` directory:

- **BBO Capstone Datasheet** — documents the motivation, composition,
  collection process, preprocessing, intended uses, distribution and
  maintenance of the optimisation dataset.

- **BBO Model Card** — documents the GP-UCB optimisation approach, intended
  use, ten-round strategy, performance, assumptions, limitations, transparency
  and reproducibility considerations.

---

## Limitations

Important limitations include:

- a small evaluation budget;
- increasingly sparse coverage as dimensionality increases;
- dependence on GP covariance assumptions;
- non-uniform sampling produced by the acquisition strategy;
- possible narrow or unexplored optima; and
- uncertainty about the response surface outside evaluated regions.

Consequently, the best observed solution should not be interpreted as a
certificate of global optimality.

---

## Future Improvements

Possible extensions include:

- comparison with Expected Improvement (EI);
- comparison with Probability of Improvement (PI);
- additional GP kernel and hyperparameter studies;
- automated surrogate-model diagnostics;
- alternative strategies for higher-dimensional functions;
- batch Bayesian optimisation; and
- application to industrial optimisation problems such as refinery operation,
  process design and simulation-based optimisation.

---

## Project Documentation Status

The repository currently contains the project README, dataset datasheet and
optimisation model card. Source code, complete observations, initial datasets,
figures and reproducibility files will be added progressively.
