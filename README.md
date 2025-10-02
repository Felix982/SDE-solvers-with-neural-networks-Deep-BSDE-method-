# SDE-solvers-with-neural-networks-Deep-BSDE-method-
This repository contains my **PyTorch implementation of the Deep BSDE method** for solving high-dimensional **stochastic differential equations (SDEs)** and related **partial differential equations (PDEs)**, following the approach of [Han, Jentzen & E (2017), *Deep Learning-Based Numerical Methods for High-Dimensional Parabolic PDEs and BSDEs*].

### Highlights
- Full implementation of **custom autograd operators** for recursive BSDE dynamics (forward & backward).
- **Time-dependent neural networks** (time-embedded MLPs) to parameterize control variables.
- Applications to:
  - **Black–Scholes option pricing** (financial PDEs).
  - **Hamilton–Jacobi–Bellman (HJB) equations** (control theory).
  - Additional nonlinear drivers from the paper.
- Includes **training loops, evaluation metrics (RMSE, correlation, relative error), and plotting utilities**.

### Why this project?
I had a lot of free time before the start of my M2 (MVA) and I wanted to work with SDE's (I love stochastic calculus) and solidify my torch skills
