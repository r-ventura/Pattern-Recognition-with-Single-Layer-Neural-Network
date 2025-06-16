# 🔢 Digit Recognition with Single-Layer Neural Networks

Optimization algorithms (GM, BFGS, SGM) for digit pattern recognition using MATLAB. Comparative analysis of convergence and accuracy on blurred digit datasets.

## 📌 Project Context

Developed for the *Mathematical Optimization* course in the **Bachelor's Degree in Data Science and Engineering** at **Universitat Politècnica de Catalunya (UPC)**, this project implements and compares optimization methods for training a single-layer neural network (SLNN) to recognize handwritten digits.

## 📊 Problem Definition
- **Task:** Classify blurred digits (0-9) using SLNN
- **Key Metrics:**
  - Loss function minimization (𝐿̃)
  - Test accuracy (Accuracy<sup>TE</sup>)
- **Constraints:** 
  - Training set: 20,000 samples 
  - Test set: 2,000 samples
  - Regularization parameter λ ∈ {0, 0.01, 0.1}

## ⚙️ Methodology
Three optimization methods were implemented and compared:

1. **Gradient Method (GM)**
   - Basic first-order optimization
   - Line search for step size (α)

2. **BFGS (Quasi-Newton Method)**
   - Approximates Hessian matrix
   - Faster convergence than GM

3. **Stochastic Gradient Method (SGM)**
   - Random direction selection
   - Computationally efficient per iteration

## 🔍 Key Findings
- **Convergence Analysis:**
  - SGM showed fastest local convergence (lowest tex/niter)
  - BFGS achieved best global convergence (100% cases)
  - Optimal λ = 0.01 balances accuracy and convergence

- **Accuracy Results:**
  - Best test accuracy: **99.8%** (SGM with λ=0)
  - Digit "8" most challenging (97.0% accuracy)
  - Digits "1", "4", and "7" achieved 100% accuracy

## 📂 Repository Structure
