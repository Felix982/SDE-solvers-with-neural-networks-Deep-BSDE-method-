# SDE-solvers-with-neural-networks-Deep-BSDE-method-
This repository contains my **PyTorch implementation of the Deep BSDE method** for solving high-dimensional **stochastic differential equations (SDEs)** and related **partial differential equations (PDEs)**, following the approach of [Han, Jentzen & E (2017), *Deep Learning-Based Numerical Methods for High-Dimensional Parabolic PDEs and BSDEs*].

### 🌟 Highlights
- Full implementation of **custom autograd operators** for recursive BSDE dynamics (forward & backward).
- **Time-dependent neural networks** (time-embedded MLPs) to parameterize control variables.
- Applications to:
  - **Black–Scholes option pricing** (financial PDEs).
  - **Hamilton–Jacobi–Bellman (HJB) equations** (control theory).
  - Additional nonlinear drivers from the paper.
- Includes **training loops, evaluation metrics (RMSE, correlation, relative error), and plotting utilities**.
- Optimized for **GPU training with checkpointing and mixed precision**.

### 🚀 Why this project?
This was part of my Master’s research work, where I translated a cutting-edge mathematical method into working code. It combines:
- **Probability & SDE theory**
- **Numerical analysis**
- **Deep learning (PyTorch)**  
and demonstrates the interplay between mathematics, finance, and AI.

### 📊 Example results
*(Add one or two convergence plots here from your training, e.g. Black–Scholes u0 ≈ 67 with training curves)*

### 🔧 Usage
```bash
pip install -r requirements.txt
python src/training_bs.py
