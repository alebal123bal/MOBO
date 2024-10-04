# MOBO
Simple MultiObjective Bayesian Optimization with three foo functions

This repository contains examples of Bayesian Optimization (BO) implemented using [BoTorch](https://botorch.org/) and [PyTorch](https://pytorch.org/). The examples demonstrate both **Single-Objective** and **Multi-Objective** Bayesian Optimization on simple mathematical functions. These scripts serve as educational tools for understanding the fundamentals of Bayesian Optimization and its application to optimization problems.

## Table of Contents

- [Features](#features)
- [Files Overview](#files-overview)
- [Installation](#installation)
- [Usage](#usage)
  - [Single-Objective Optimization](#single-objective-optimization)
  - [Multi-Objective Optimization (MOBO)](#multi-objective-optimization-mobo)
- [Dependencies](#dependencies)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgements](#acknowledgements)

## Features

- **Single-Objective Bayesian Optimization**: Optimize a simple quadratic function using BoTorch.
- **Multi-Objective Bayesian Optimization (MOBO)**: Optimize multiple functions simultaneously, demonstrating Pareto front identification.
- **Visualization Support**: Plotting capabilities using Matplotlib to visualize optimization progress and results.
- **Reproducibility**: Fixed random seeds ensure consistent results across runs.

## Files Overview

### 1. `custom_SOBO.py`

**Description**:  
This script demonstrates how to perform Single-Objective Bayesian Optimization to find the minimum of the function \( f(x, y) = (x-3)^2 + (y-3)^2 \). The optimization is conducted within a defined domain, and the progress is logged at each iteration.

**Key Components**:
- Defines the objective function to be minimized.
- Initializes the Gaussian Process (GP) model.
- Iteratively fits the model, selects new points using the acquisition function, and updates the model with new evaluations.
- (Optional) Contains plotting code to visualize the optimization process.

### 2. `custom_MOBO.py`

**Description**:  
This script showcases Multi-Objective Bayesian Optimization (MOBO) by optimizing three objective functions simultaneously:
1. \( f_1(x) = (x_1 - 3)^2 + 1 \)
2. \( f_2(x) = x_1^2 + x_2^2 \)
3. \( f_3(x) = \sqrt{|x_1 + x_2 + 1|} \)

The goal is to identify the Pareto front—a set of non-dominated solutions representing the trade-offs between objectives.

**Key Components**:
- Defines multiple objective functions to be minimized.
- Initializes a multi-output GP model.
- Utilizes the Expected Hypervolume Improvement (EHVI) acquisition function for selecting new points.
- Iteratively updates the model and logs the optimization progress.
- Performs Pareto analysis to identify optimal points across objectives.
- (Optional) Contains plotting code for visualization.

## Installation

1. **Clone the Repository**:
    ```bash
    git clone https://github.com/alebal123bal/MOBO.git
    cd MOBO
    ```

2. **Create a Virtual Environment** (optional but recommended):
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3. **Install Dependencies**:

    ```bash
    pip install torch botorch gpytorch matplotlib numpy
    ```

## Usage

### Single-Objective Optimization

To run the single-objective Bayesian Optimization example:

```bash
python custom_SOBO.py
```

**Description**:  
This script will perform Bayesian Optimization to find the minimum of the function \( f(x, y) = (x-3)^2 + (y-3)^2 \). It initializes with random points, iteratively selects new points based on the acquisition function, and updates the GP model. The optimization progress is printed to the console, and plotting code is available (commented out) for visualization.

### Multi-Objective Optimization (MOBO)

To run the multi-objective Bayesian Optimization example:

```bash
python custom_MOBO.py
```

**Description**:  
This script performs MOBO on three objective functions. It aims to identify a set of optimal trade-off solutions (Pareto front). The script initializes with random points, uses the EHVI acquisition function to select new points, updates the GP models, and logs the optimization progress. Pareto analysis results are printed at the end, and plotting code is available (commented out) for visualization.

## Dependencies

- **Python**: 3.7 or higher
- **PyTorch**: For tensor computations and model definitions.
- **BoTorch**: For Bayesian Optimization functionalities.
- **GPyTorch**: As a backend for Gaussian Process models.
- **Matplotlib**: For plotting and visualization.
- **NumPy**: For numerical operations.

*Ensure that you have a compatible CUDA version if you plan to run the scripts on a GPU. Otherwise, the scripts will default to CPU execution.*

## Results

Upon running the scripts, you will observe iterative logs detailing the optimization progress. For the multi-objective example, the final Pareto front analysis identifies points that optimize each objective individually.

*Example Output for Multi-Objective BO*:
```
Iteration 1:
  New input point: [x, y]
  New objectives (negated): [f1, f2, f3]
  Acquisition function value: value

...

Final Pareto Frontier Analysis:
  Point that optimizes Objective 1: (x, y) = [x1, y1] with objectives = [f1, f2, f3]
  Point that optimizes Objective 2: (x, y) = [x2, y2] with objectives = [f1, f2, f3]
  Point that optimizes Objective 3: (x, y) = [x3, y3] with objectives = [f1, f2, f3]
```

*Plotting*:  
While plotting code is included in both scripts (currently commented out), you can enable it by uncommenting the relevant sections to visualize the optimization path and Pareto front.

## Contributing

Contributions are welcome! If you find any bugs or have suggestions for improvements, please open an issue or submit a pull request.

## License

This project is licensed under the [MIT License](LICENSE).

## Acknowledgements

- [BoTorch](https://botorch.org/) for providing a powerful library for Bayesian Optimization.
- [PyTorch](https://pytorch.org/) for its efficient tensor computation capabilities.
- The open-source community for their invaluable resources and support.

---

*Happy Optimizing!*