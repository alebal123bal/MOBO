"""
MOBO optimization of simple functions.
import matplotlib.pyplot as plt
"""

import torch
from botorch.models import SingleTaskGP
from botorch.models.model_list_gp_regression import ModelListGP
from botorch.acquisition.multi_objective.logei import qLogExpectedHypervolumeImprovement
from botorch.optim import optimize_acqf
from botorch.fit import fit_gpytorch_mll
from botorch.utils.multi_objective.pareto import is_non_dominated
from botorch.utils.transforms import normalize, unnormalize
from botorch.utils.multi_objective.box_decompositions.non_dominated import FastNondominatedPartitioning
from botorch.sampling import SobolQMCNormalSampler
from gpytorch.mlls.sum_marginal_log_likelihood import SumMarginalLogLikelihood

# Set random seed for reproducibility
torch.manual_seed(42)


# Define the objective functions (negated for minimization)
def objective(x):
    """
    Return the 3 objective functions to be minimized.\n

    Args:\n
        X: Input tensor\n

    Returns:\n
        torch.stack with correct dim and functions
    """
    # (x - 3)**2 + 1 has infinite sets of minima at (3.0, _any_)
    f1 = (x[:, 0] - 3) ** 2 + 1

    # x**2 + y**2 has a unique minimum at (0, 0)
    f2 = x[:, 0] ** 2 + x[:, 1] ** 2

    # sqrt(|x + y + 1|) has infinite sets of minima where x + y = -1
    f3 = torch.sqrt(torch.abs(x[:, 0] + x[:, 1] + 1))

    # Negate functions for trating a minimization as a maximization problem
    return torch.stack([-f1, -f2, -f3], dim=-1)


# Set up the optimization
DTYPE = torch.float64
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define bounds
bounds = torch.tensor([[-5.0, -5.0], [5.0, 5.0]], dtype=DTYPE, device=DEVICE)

# Generate initial training data
my_in_points = bounds[0] + (bounds[1] - bounds[0]) * torch.rand(20, 2, dtype=DTYPE, device=DEVICE)
my_out_points = objective(my_in_points)


# Define model
def initialize_model(in_points, out_points):
    """
    Initialize the ModeListGp.\n

    Args:\n
        in_points: just train_x\n
        out_points: just train_yßn

    Returns:\n
        mll and model
    """
    # Normalize inputs
    in_points_normalized = normalize(in_points, bounds=bounds)
    # Define models for each objective
    models = []
    for i in range(out_points.shape[-1]):
        models.append(SingleTaskGP(in_points_normalized, out_points[..., i : i + 1], likelihood=None))
    model = ModelListGP(*models)
    mll = SumMarginalLogLikelihood(model.likelihood, model)
    return mll, model


mll, model = initialize_model(my_in_points, my_out_points)

# Plotting setup
# plt.ion()
# fig = plt.figure(figsize=(8, 6))
# ax_3d = fig.add_subplot(111, projection="3d")

# Optimization loop
MAX_N = 20
for iteration in range(MAX_N):
    # Basically, if the MLL is low, it suggests the model isn't fitting the data well,
    # so we might "smell a rat."
    # Fit the model
    fit_gpytorch_mll(mll, options={"disp": False})

    # Dynamically estimate the reference point based on the worst observed values
    worst_observed_values = torch.min(my_out_points, dim=0).values
    ref_point = worst_observed_values - 0.1  # Keep ref_point as a tensor

    # Create a partitioning of the objective space using the reference point and observed outcomes.
    # This is used to efficiently calculate the hypervolume for multi-objective optimization
    # and to find dominated (all points are worse than some point on the Pareto front )
    # and non dominated regions (the good region)
    partitioning = FastNondominatedPartitioning(ref_point=ref_point, Y=my_out_points)

    # Since calculating EHVI involves estimating expected values over uncertain outcomes (usually calcing an integral),
    # this sampler helps to approximate these expectations via Quasi-Monte Carlo (QMC) sampling.
    # Analytical solutions for these expectations  (integrals) are typically infeasible, so QMC sampling
    # provides an efficient and accurate approximation.
    # Create the SobolQMCNormalSampler with the correct sample_shape and optional seed
    sampler = SobolQMCNormalSampler(sample_shape=torch.Size([128]), seed=iteration)

    # This acquisition function aims to improve one objective function without deteriorating the others
    # i.e. it tries to expand the Pareto frontier finding a better tradeoff between the 3 objective functions
    # Ensure there are no pending points (X_pending), and use default eta
    qEHVI = qLogExpectedHypervolumeImprovement(
        model=model,
        ref_point=ref_point,  # Use the dynamically estimated reference point as a tensor
        partitioning=partitioning,
        sampler=sampler,  # Use the correctly defined SobolQMCNormalSampler here
        objective=None,  # Default to IdentityMCMultiOutputObjective
        constraints=None,  # No constraints in this case
        X_pending=None,  # No pending points
        eta=0.1,  # Increase eta for more exploration
    )

    # Do Optimize the acquisition function: obtain new candidate to explore and see how big the acq value is.
    # acq_value variable tells how much I expect to improve my function knowledge.
    # new_x is either a potential extreme or located in a very dark area.
    new_in_points, acq_value = optimize_acqf(
        acq_function=qEHVI,
        bounds=torch.stack([torch.zeros(2, dtype=DTYPE, device=DEVICE), torch.ones(2, dtype=DTYPE, device=DEVICE)]),
        q=1,
        num_restarts=10,
        raw_samples=512,  # Increase raw samples for better optimization
        options={"maxiter": 200, "disp": False},
    )

    # Unnormalize the new point to correctly calc the objective outputs
    new_in_points = unnormalize(new_in_points, bounds=bounds)

    # Evaluate the new point
    new_out_points = objective(new_in_points)

    # Update training points
    my_in_points = torch.cat([my_in_points, new_in_points])
    my_out_points = torch.cat([my_out_points, new_out_points])

    # Update the model
    mll, model = initialize_model(my_in_points, my_out_points)

    # Detailed logging for each iteration
    print(f"Iteration {iteration + 1}:")
    print(f"  New input point: {new_in_points.squeeze().tolist()}")
    print(f"  New objectives (negated): {new_out_points.squeeze().tolist()}")
    print(
        f"  Acquisition function value: {acq_value.item()}"
    )  # This aims to be as big as possible, but decreases over time (there is more knowledge)


# Final Pareto analysis
pareto_mask = is_non_dominated(my_out_points)
pareto_points = my_in_points[pareto_mask]
pareto_objectives = my_out_points[pareto_mask]

# Find the points on the Pareto front that optimize each objective
optimal_f1_idx = torch.argmax(pareto_objectives[:, 0])
optimal_f2_idx = torch.argmax(pareto_objectives[:, 1])
optimal_f3_idx = torch.argmax(pareto_objectives[:, 2])

optimal_f1_point = pareto_points[optimal_f1_idx]
optimal_f2_point = pareto_points[optimal_f2_idx]
optimal_f3_point = pareto_points[optimal_f3_idx]

# Logging the final results
print("\nFinal Pareto Frontier Analysis:")
print(
    f"  Point that optimizes Objective 1: (x, y) = {optimal_f1_point.tolist()} with objectives = {(-pareto_objectives[optimal_f1_idx]).tolist()}"
)
print(
    f"  Point that optimizes Objective 2: (x, y) = {optimal_f2_point.tolist()} with objectives = {(-pareto_objectives[optimal_f2_idx]).tolist()}"
)
print(
    f"  Point that optimizes Objective 3: (x, y) = {optimal_f3_point.tolist()} with objectives = {(-pareto_objectives[optimal_f3_idx]).tolist()}"
)
