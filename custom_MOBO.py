import torch
import matplotlib.pyplot as plt
from botorch.models import SingleTaskGP
from botorch.acquisition.multi_objective import qExpectedHypervolumeImprovement
from botorch.optim import optimize_acqf
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls.sum_marginal_log_likelihood import SumMarginalLogLikelihood
from botorch.models.model_list_gp_regression import ModelListGP
from botorch.utils.multi_objective.box_decompositions.non_dominated import FastNondominatedPartitioning
from botorch.utils.transforms import normalize, unnormalize
from botorch.sampling import SobolQMCNormalSampler
from botorch.utils.multi_objective.pareto import is_non_dominated

# Set random seed for reproducibility
torch.manual_seed(42)

# Define the objective functions (negated for minimization)
def objective(X):
    # (x - 3)**2 + 1 has a inf sets of minimums at (3.0, _any_)
    f1 = ((X[:, 0] - 3)**2 + 1)
    
    # x**2 + y**2 has a unique minimum at (0, 0)
    f2 = (X[:, 0]**2 + X[:, 1]**2)

    # sqrt(|x + y + 1|) has a inf sets of minimums where x + y = -1
    f3 = torch.sqrt(torch.abs(X[:, 0] + X[:, 1] + 1))

    #Negate functions for minimization problem
    return torch.stack([-f1, -f2, -f3], dim=-1)

# Set up the optimization
dtype = torch.float64
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define bounds
bounds = torch.tensor([[-5.0, -5.0], [5.0, 5.0]], dtype=dtype, device=device)

# Generate initial training data
my_in_points = bounds[0] + (bounds[1] - bounds[0]) * torch.rand(20, 2, dtype=dtype, device=device)
my_out_points = objective(my_in_points)

# Define model
def initialize_model(in_points, out_points):
    # Normalize inputs
    train_x_normalized = normalize(in_points, bounds=bounds)
    # Define models for each objective
    models = []
    for i in range(out_points.shape[-1]):
        models.append(SingleTaskGP(train_x_normalized, out_points[..., i:i+1], likelihood=None))
    model = ModelListGP(*models)
    mll = SumMarginalLogLikelihood(model.likelihood, model)
    return mll, model

mll, model = initialize_model(my_in_points, my_out_points)

# Define reference point for hypervolume calculation
# Adjusted reference point considering all three objectives
ref_point = torch.tensor([-3.0, -3.0, -2.0], dtype=dtype, device=device)

# Plotting setup
plt.ion()
fig = plt.figure(figsize=(8, 6))
ax_3d = fig.add_subplot(111, projection='3d')

# Optimization loop
n_iterations = 20
for iteration in range(n_iterations):
    # Fit the model
    fit_gpytorch_mll(mll, options={"disp": False})

    # Define the acquisition function
    partitioning = FastNondominatedPartitioning(ref_point=ref_point, Y=my_out_points)
    
    # Create the SobolQMCNormalSampler with the correct sample_shape and optional seed
    sampler = SobolQMCNormalSampler(sample_shape=torch.Size([128]), seed=iteration)

    # Ensure there are no pending points (X_pending), and use default eta
    qEHVI = qExpectedHypervolumeImprovement(
        model=model,
        ref_point=ref_point.tolist(),  # Ensure the ref_point is a list, as per the docstring
        partitioning=partitioning,
        sampler=sampler,  # Use the correctly defined SobolQMCNormalSampler here
        objective=None,  # Default to IdentityMCMultiOutputObjective
        constraints=None,  # No constraints in this case
        X_pending=None,  # No pending points
        eta=0.1  # Increase eta for more exploration
    )

    # Optimize the acquisition function
    new_in_points, acq_value = optimize_acqf(
        acq_function=qEHVI,
        bounds=torch.stack([torch.zeros(2, dtype=dtype, device=device), torch.ones(2, dtype=dtype, device=device)]),
        q=1,
        num_restarts=20,
        raw_samples=512,  # Increase raw samples for better optimization
        options={"maxiter": 1000, "disp": False}
    )

    # Unnormalize the new point
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
    print(f"  Acquisition function value: {acq_value.item()}")

    # Plotting the current Pareto front
    ax_3d.clear()

    # 3D Pareto front visualization: Objective 1 vs. Objective 2 vs. Objective 3
    ax_3d.scatter(-my_out_points[:, 0].cpu().numpy(), -my_out_points[:, 1].cpu().numpy(),
                  -my_out_points[:, 2].cpu().numpy(), color='blue', label='Sampled Points')
    ax_3d.scatter(-new_out_points[:, 0].cpu().numpy(), -new_out_points[:, 1].cpu().numpy(),
                  -new_out_points[:, 2].cpu().numpy(), color='red', label='New Sample', edgecolor='black', s=100)
    ax_3d.set_xlabel('Objective 1')
    ax_3d.set_ylabel('Objective 2')
    ax_3d.set_zlabel('Objective 3')
    ax_3d.set_title(f'3D Pareto Front - Iteration {iteration + 1}')
    ax_3d.legend()

    plt.draw()
    plt.pause(0.5)

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
print(f"  Point that optimizes Objective 1: (x, y) = {optimal_f1_point.tolist()} with objectives = {(-pareto_objectives[optimal_f1_idx]).tolist()}")
print(f"  Point that optimizes Objective 2: (x, y) = {optimal_f2_point.tolist()} with objectives = {(-pareto_objectives[optimal_f2_idx]).tolist()}")
print(f"  Point that optimizes Objective 3: (x, y) = {optimal_f3_point.tolist()} with objectives = {(-pareto_objectives[optimal_f3_idx]).tolist()}")

plt.ioff()
plt.show()