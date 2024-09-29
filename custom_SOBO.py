"""
Example: find minimum of x**2 + y**2 using Bayesian Optimization.\n
Dominium is defined as [-SIDE, SIDE][-SIDE, SIDE] and the solution will be x = y = 0.\n
"""

import torch
import matplotlib.pyplot as plt
import numpy as np
from botorch.models import SingleTaskGP
from botorch.models.transforms.input import Normalize
from botorch.models.transforms.outcome import Standardize
from botorch.fit import fit_gpytorch_mll
from botorch.optim import optimize_acqf
from botorch.acquisition import LogExpectedImprovement
from gpytorch.mlls import ExactMarginalLogLikelihood


# Define objective function
def objective(x_in):
    """Returns the objective function to be minimized.\n
    To be replaced by a function evaluation directly from ADE.

    Args:
        x: pytorch tensor containing inputs (x and y in this case)

    Returns:
        lambda (x-3)**2 + (y-3)**2
    """
    x, y = x_in[:, 0], x_in[:, 1]
    return -((x - 3) ** 2 + (y - 3) ** 2).unsqueeze(-1)


# Define device and numerical repr.
DTYPE = torch.float64
DEVICE = "cpu"

# Define bounds (it is a square)
SIDE = 5.0
bounds = torch.tensor([[-SIDE, -SIDE], [SIDE, SIDE]], dtype=DTYPE, device=DEVICE)

# Generate initial training data
train_x = torch.rand(int(SIDE) * 5, 2, dtype=DTYPE) * 2 * SIDE - SIDE
train_y = objective(train_x)

train_x_dim = train_x.shape[-1]

# Define model
model = SingleTaskGP(train_X=train_x, train_Y=train_y, input_transform=Normalize(d=train_x_dim))
mll = ExactMarginalLogLikelihood(model.likelihood, model)

# Save the iterations and guesses to later plot with matplotlib
iterations = [train_x.numpy()]
values = [train_y.numpy()]

# Loop through
MAX_N = 40

for i in range(MAX_N):
    # Fit the model
    fit_gpytorch_mll(mll)

    # Define intelligent acquisition function
    EI = LogExpectedImprovement(model, best_f=train_y.max(), maximize=True)

    # Do optimize acquisition function
    new_x, acq_value = optimize_acqf(
        EI,
        bounds,
        q=1,
        num_restarts=10,
        raw_samples=2048,
        options={
            "maxiter": 2000,
            "disp": False,
            "ftol": 1e-8,  # (default 1e-5)
            "gtol": 1e-8,  # (default 1e-5)
            "maxcor": 20,  # (default 10)
        },
    )

    # Eval the new point
    new_y = objective(new_x)

    # Update training points
    train_x = torch.cat([train_x, new_x])
    train_y = torch.cat([train_y, new_y])

    train_x_dim = train_x.shape[-1]
    train_y_dim = train_y.shape[-1]

    # Update the model
    model = SingleTaskGP(train_x, train_y, input_transform=Normalize(d=train_x_dim), outcome_transform=Standardize(m=train_y_dim))
    mll = ExactMarginalLogLikelihood(model.likelihood, model)

    # Store new values
    iterations.append(new_x.numpy())
    values.append(new_y.numpy())

    print(f"Iteration number {i} has the following params:")
    print(f"X (input): {new_x}")
    print(f"Y (objective): {new_y}")
    print(f"Acquisition value: {acq_value}\n")

# Plot everything
# x = np.linspace(-SIDE, SIDE, int(SIDE) * 200)
# y = np.linspace(-SIDE, SIDE, int(SIDE) * 200)
# X, Y = np.meshgrid(x, y)
# Z = X**2 + 2 * Y**2

# plt.figure(figsize=(10, 6))

# plt.contourf(X, Y, Z, levels=50, cmap="viridis")
# plt.colorbar(label="objecive function value")

# my_x_plt = [x[:, 0] for x in iterations]
# my_y_plt = [y[:, 1] for y in iterations]

# my_x_plt = np.concatenate((np.array([e for e in my_x_plt[0]]), [a[0] for a in my_x_plt[1:]]))
# my_y_plt = np.concatenate((np.array([e for e in my_y_plt[0]]), [a[0] for a in my_y_plt[1:]]))

# plt.plot(my_x_plt, my_y_plt, "ro-", label="iteration")
# plt.xlabel("x")
# plt.ylabel("y")
# plt.title("My guesses title")
# plt.legend()
# plt.grid(True)
# plt.show()

# Save iterations and pick best one

print(values)
