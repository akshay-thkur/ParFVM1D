# ParFVM1D - Parallel One-dimensional Finite Volume Solver for Hyperbolic Conservation Laws

This repository contains a Julia implementation of a **Parallel Finite Volume Method (FVM)** solver, developed as the final project for **ACMS 60212: Advanced Scientific Computing** at the University of Notre Dame by me and Alexander Perez. 

The framework is designed to solve one-dimensional, time-dependent partial differential equations (PDEs) using distributed computing via **MPI**.

## Theoretical Overview

The solver addresses the general form of a system of $m$ conservation laws:

$$\frac{\partial q}{\partial t} + \frac{\partial f(q)}{\partial x} = S(q)$$

### Implemented Physics
1.  **Inviscid Burgers’ Equation:** A fundamental model for nonlinear advection and shock wave formation where $f(u) = \frac{u^2}{2}$.
2.  **Shallow Water Equations (SWE):** Models incompressible, inviscid fluid flow subject to gravitational forces, conserving mass (fluid height $\rho$) and momentum ($\rho v$).



---

## Features

* **MPI Parallelization:** Scalable 1D domain decomposition using `MPI.jl`.
* **Ghost Cell Communication:** Efficient synchronization of subdomains using non-blocking communication (`Isend`/`Irecv`).
* **Riemann Solvers:**
    * **Godunov’s Method:** An exact Riemann solver approach used for the Burgers' equation.
    * **Roe’s Approximate Riemann Solver:** A linearized solver used for the Shallow Water Equations.
* **Vectorized Performance:** Optimized flux calculations using `Tullio.jl` and `LinearAlgebra`.



---

## Project Structure

| File | Description |
| :--- | :--- |
| `addpackages.jl` | Setup script for the Julia environment and dependencies. |
| `mpi_solve_burgers_ex1.jl` | Parallel Burgers' solver with a square-pulse initial condition. |
| `mpi_solve_burgers_ex2.jl` | Parallel Burgers' solver with a step-function initial condition. |
| `mpi_solve_swe_ex1.jl` | Parallel Shallow Water Equation solver utilizing the Roe flux formulation. |
---

## Getting Started
* Required Julia Packages: `MPI`, `Tullio`, `MAT`, `LinearAlgebra`

### Installation
Initialize the environment and download dependencies:
```bash
julia addpackages.jl
