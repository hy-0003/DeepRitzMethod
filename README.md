# Deep Ritz Method: Solving High-Dimensional PDEs via Deep Learning

## Overview

This repository contains the implementation and numerical analysis of the **Deep Ritz Method (DRM)**, a deep learning-based framework for solving Partial Differential Equations (PDEs).

The project conducts a comparative study between the traditional **Finite Element Method (FEM)** and DRM. While both share the same theoretical foundation rooted in the **Ritz variational principle**, DRM replaces mesh-based approximations with deep neural networks. This evolution allows DRM to overcome the "curse of dimensionality" that plagues traditional methods.

**Key Highlights:**
- **Theoretical Analysis:** Explores the connection between FEM and DRM.
- **Mesh-Free Solver:** Eliminates the need for complex mesh generation.
- **High-Dimensional Capability:** Successfully solves a **10-dimensional Poisson equation** with a relative error of **0.65%**, whereas standard FEM failed due to memory exhaustion (OOM).

## Repository Structure

This repository consists of Code folder and Report folder.

## Acknowledgements

Special thanks to **Zeyu Jia**, **Dinghuai Zhang**, and **Zhengming Zou** from the School of Mathematical Sciences at Peking University. Their open-source project provided a robust foundation for this work. We extensively utilized their LaTeX template, insightful report, and codebase as valuable references for our implementation of the Deep Ritz Method.

**Reference Repository:** [ZeyuJia/DeepRitzMethod](https://github.com/ZeyuJia/DeepRitzMethod)
