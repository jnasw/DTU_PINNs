from src.functions import *
import torch
import wandb
import hydra
from src.dataset.create_dataset_functions import ODE_modelling
from src.ode.sm_models_d import SynchronousMachineModels
import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


os.environ["HYDRA_FULL_ERROR"]="1"

def plot_trajectory_from_oderesult(solution_all, var_names=None, sample_idx=0):
    """
    Plot a trajectory from a list of OdeResult objects returned by solve_ivp.

    Parameters
    ----------
    solution_all : list of OdeResult
        Each entry is a SciPy OdeResult (with .t and .y attributes).
    var_names : list of str, optional
        Names of the variables. If None, generic names are used.
    sample_idx : int
        Which trajectory (initial condition) to plot.
    """
    sol = solution_all[sample_idx]   # an OdeResult
    t_array = sol.t                  # time points (1D)
    y = sol.y                        # shape: [n_vars, n_timepoints]

    n_vars = y.shape[0]
    if var_names is None:
        var_names = [f"Var {i}" for i in range(n_vars)]

    plt.figure(figsize=(10, 5))
    for i in range(n_vars):
        plt.plot(t_array, y[i, :], label=var_names[i])

    plt.title(f"Trajectory for Sample #{sample_idx}")
    plt.xlabel("Time")
    plt.ylabel("State Variables")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_selected_variable(solution_all, t_vector, var_names, var_name, n_samples=5, start_idx=0):
    """
    Plot a chosen state variable for several trajectories.

    Parameters
    ----------
    solution_all : list of OdeResult
        List of SciPy OdeResult objects returned by solve_ivp.
    t_vector : np.ndarray
        Time vector (e.g., solution_all[0].t)
    var_names : list of str
        Names of the state variables in the same order as solution_all[i].y
    var_name : str
        Name of the variable to plot (must be in var_names)
    n_samples : int
        How many trajectories to plot at once
    start_idx : int
        Index of the first trajectory to plot (default 0)
    """
    # Index of chosen variable
    if var_name not in var_names:
        raise ValueError(f"{var_name} not in var_names")
    var_idx = var_names.index(var_name)

    plt.figure(figsize=(10, 5))
    for i in range(start_idx, min(start_idx + n_samples, len(solution_all))):
        sol = solution_all[i]
        y = sol.y  # shape [n_vars, n_timepoints]
        plt.plot(t_vector, y[var_idx, :], label=f"Sample {i}")

    plt.title(f"{var_name} for Samples {start_idx} to {start_idx + n_samples - 1}")
    plt.xlabel("Time")
    plt.ylabel(var_name)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

def add_noise_to_solution(solution_all, noise_std=0.01):
    import copy
    noisy_all = []
    for sol in solution_all:
        noisy_sol = copy.deepcopy(sol)
        noisy_sol.y = sol.y + np.random.normal(0, noise_std, sol.y.shape)
        noisy_all.append(noisy_sol)
    return noisy_all

def plot_ic_distribution(init_conditions, var_names, title="Initial Conditions"):
    ic_array = np.array(init_conditions)
    df = pd.DataFrame(ic_array, columns=var_names)

    # Drop constant variables (std == 0)
    df = df.loc[:, df.std() > 1e-5]

    sns.pairplot(df, corner=True, plot_kws={"s": 15, "alpha": 0.7})
    plt.suptitle(title, y=0.95, fontsize=16)
    plt.tight_layout()
    plt.show()

def plot_ic_distributions_combined(ic_sets, var_names, titles):

    n_sets = len(ic_sets)
    n_vars = len(var_names)

    fig, axs = plt.subplots(
        n_sets, n_vars,
        figsize=(4 * n_vars, 2.3 * n_sets),
        constrained_layout=True
    )

    for i, ic_set in enumerate(ic_sets):
        ic_array = np.array(ic_set)

        for j, var in enumerate(var_names):
            ax = axs[i, j] if n_sets > 1 else axs[j]
            data = ic_array[:, j]

            min_val, max_val = np.min(data), np.max(data)
            is_constant = np.isclose(min_val, max_val)

            if is_constant:
                # Draw a single vertical line with annotation
                ax.axvline(min_val, color='black', linewidth=2)
                ax.set_xlim(min_val - 0.01, max_val + 0.01)
                ax.set_yticks([])
                ax.set_xticks([round(min_val, 2)])
            else:
                # Use smart bin count (unique or FD rule)
                unique_vals = np.unique(data)
                bins = len(unique_vals) if len(unique_vals) < 20 else 'auto'
                ax.hist(data, bins=bins, color='cornflowerblue', edgecolor='black', alpha=0.85)

                # Add margin
                margin = (max_val - min_val) * 0.1
                ax.set_xlim(min_val - margin, max_val + margin)

            # Titles & labels
            if i == 0:
                ax.set_title(var, fontsize=10)
            if j == 0:
                ax.set_ylabel(titles[i], fontsize=10)

            ax.tick_params(axis='both', labelsize=8)
            ax.grid(True, linestyle=':', linewidth=0.5)

    fig.suptitle("Initial Condition Distribution Across Sampling Strategies", fontsize=14)
    plt.show()

def trajectory_statistics(solution_all, var_names):

    stat_dict = {name: [] for name in var_names}
    for sol in solution_all:
        for i, name in enumerate(var_names):
            stat_dict[name].append(np.std(sol.y[i]))

    df = pd.DataFrame(stat_dict)
    return df.describe()


def compute_ode_residual(solution, ode_func):
    """
    Compute the mean squared residual for a trajectory from solve_ivp.

    Parameters
    ----------
    solution : OdeResult
        Output from scipy.integrate.solve_ivp (has .t and .y)
    ode_func : callable
        The RHS function of the ODE, as used in solve_ivp: f(t, x)

    Returns
    -------
    float
        Mean squared residual over all time steps
    """
    t = solution.t                 # shape: [n_timesteps]
    x = solution.y.T               # shape: [n_timesteps, n_vars]
    
    # Step 1: numerical derivative dx/dt from solver output
    dxdt_numeric = np.gradient(x, t, axis=0)  # shape: [n_timesteps, n_vars]

    residuals = []

    # Step 2: evaluate residual at each time step
    for i in range(len(t)):
        t_i = t[i]
        x_i = x[i]                            # current state vector
        dxdt_model = ode_func(t_i, x_i)       # ODE RHS evaluation
        diff = dxdt_numeric[i] - dxdt_model   # difference vector
        res_sq = np.sum(diff**2)              # squared residual
        residuals.append(res_sq)

    # Step 3: return average residual across time steps
    return np.mean(residuals)

def plot_residual_histogram(residuals, label=""):
    plt.figure(figsize=(6, 4))
    plt.hist(residuals, bins=30, alpha=0.7, edgecolor='black')
    plt.title(f"Residual Distribution: {label}")
    plt.xlabel("Mean Squared Residual")
    plt.ylabel("Count")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_residual_histograms_combined(residual_sets, labels, colors=None):
    """
    Plot combined residual histograms for multiple sets with mean indicators.

    Parameters
    ----------
    residual_sets : list of arrays
        Residual arrays for each dataset.
    labels : list of str
        Labels for each dataset.
    colors : list of str, optional
        Colors for each dataset.
    """
    if colors is None:
        colors = ['royalblue', 'darkorange', 'forestgreen']

    plt.figure(figsize=(8, 5))

    # Common bin edges for fair comparison
    bins = np.linspace(0, max(np.max(r) for r in residual_sets), 40)

    # Plot each histogram with transparency
    for idx, (residuals, label, color) in enumerate(zip(residual_sets, labels, colors)):
        plt.hist(residuals, bins=bins, alpha=0.5, label=label, edgecolor='black', color=color)

        # Mean line
        mean_val = np.mean(residuals)
        plt.axvline(mean_val, linestyle='--', color=color, alpha=0.8)

        # Horizontal offset for annotation to avoid overlap
        offset = 0.2 * (idx - (len(residual_sets)-1)/2.0)  # shifts left/center/right
        y_pos = plt.ylim()[1] * (0.9 - 0.05*idx)          # slight vertical stagger as well
        plt.text(mean_val + offset, y_pos, f"μ={mean_val:.2f}", 
                 color=color, fontsize=9, ha='center', va='bottom')

    plt.title("Residual Distribution Comparison", fontsize=14)
    plt.xlabel("Mean Squared Residual")
    plt.ylabel("Count")
    plt.legend()
    plt.grid(True, linestyle=':', linewidth=0.7)
    plt.tight_layout()
    plt.show()


def check_high_error_focus(set4_ICs, set4_residuals, set5_ICs, top_k=200):

    set4_ICs = np.array(set4_ICs)
    set5_ICs = np.array(set5_ICs)
    set4_residuals = np.array(set4_residuals)

    # Find top-k indices
    top_k_indices = np.argsort(set4_residuals)[-top_k:]
    top_k_ICs = set4_ICs[top_k_indices]

    # Count how many top-k ICs appear in set5
    count = 0
    for ic in set5_ICs:
        if any(np.allclose(ic, top_ic) for top_ic in top_k_ICs):
            count += 1

    print(f"\n🎯 High-Error Focus Check:")
    print(f"{count}/{len(set5_ICs)} ICs in Set5 are from top-{top_k} high-residual ICs of Set4")

def plot_exploit_explore_distribution(exploit_ics, explore_ics, var_names=["theta", "omega"]):

    exploit_ics = np.array(exploit_ics)
    explore_ics = np.array(explore_ics)

    if len(var_names) < 2:
        raise ValueError("Need at least two variable names for 2D plot")

    idx_x = 0  # typically theta
    idx_y = 1  # typically omega

    plt.figure(figsize=(6, 6))
    plt.scatter(exploit_ics[:, idx_x], exploit_ics[:, idx_y], color="red", label="Exploit (High Error)", alpha=0.6)
    plt.scatter(explore_ics[:, idx_x], explore_ics[:, idx_y], color="blue", label="Explore (Random)", alpha=0.6)
    plt.xlabel(var_names[idx_x])
    plt.ylabel(var_names[idx_y])
    plt.title("Exploit vs. Explore ICs in Evo Sampling")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_trajectories_with_derivatives(solution_set3, solution_set4, solution_set5, var_names):
    t_vector = solution_set3[0].t  # use t from any solution
    fig, axs = plt.subplots(3, len(var_names), figsize=(4 * len(var_names), 9), sharex=True)
    
    sets = [solution_set3, solution_set4, solution_set5]
    set_labels = ["Set 3: Grid", "Set 4: Random/LHS", "Set 5: Evo"]

    for row in range(3):  # one row per set
        sol = sets[row][0]
        dy_dt = np.gradient(sol.y, t_vector, axis=1)

        for col, var in enumerate(var_names):
            
            ax = axs[row, col]
            ax.plot(t_vector, sol.y[col], color='tab:blue', label=var)
            ax.grid(True)

            # Create twin axis for derivative
            ax2 = ax.twinx()
            ax2.plot(t_vector, dy_dt[col], color='tab:red', linestyle='--', alpha=0.6, label=f"d{var}/dt")

            # Suppress right y-axis ticks unless it's the last column
            if col != len(var_names) - 1:
                ax2.tick_params(axis='y', labelleft=False, labelright=False)
            else:
                ax2.tick_params(axis='y', labelright=True, labelsize=8)

            if row == 0:
                ax.set_title(var, fontsize=10)

            if col == 0:
                ax.set_ylabel(set_labels[row], fontsize=10)

            # Optional: make left y-axis ticks smaller
            ax.tick_params(axis='y', labelsize=7)

    fig.suptitle("Trajectory and Derivative Comparison (First Sample from Each Set)", fontsize=14)
    plt.tight_layout()
    plt.show()

def quantify_trajectory_activity(time, trajectory):
    """Compute activity metrics for a 1D trajectory over time."""
    dt = np.gradient(time)
    dx_dt = np.gradient(trajectory, time)

    return {
        'std': np.std(trajectory),
        'range': np.max(trajectory) - np.min(trajectory),
        'l2_derivative': np.sqrt(np.sum(dx_dt**2 * dt)),
        'mean_abs_derivative': np.mean(np.abs(dx_dt)),
        'total_variation': np.sum(np.abs(np.diff(trajectory)))
    }

def compute_activity_for_all_solutions(solution_set, var_names):
    """Compute mean activity for each trajectory (averaged over variables)."""
    t = solution_set[0].t
    activity_summary = []

    for sol in solution_set:
        traj_metrics = []

        for var_idx in range(len(var_names)):
            traj = sol.y[var_idx]
            metrics = quantify_trajectory_activity(t, traj)
            traj_metrics.append(metrics['l2_derivative'])  # or choose any other metric

        # Average activity over variables
        mean_activity = np.mean(traj_metrics)
        activity_summary.append(mean_activity)

    return np.array(activity_summary)

def add_noise_to_solutions(solutions, noise_std=0.01, seed=None):
    """
    Add Gaussian noise to the trajectory data of each solution.

    Parameters
    ----------
    solutions : list of OdeResult
        List of solved trajectories (from solve_ivp).
    noise_std : float
        Standard deviation of Gaussian noise (relative to signal).
    seed : int or None
        Random seed for reproducibility.

    Returns
    -------
    noisy_solutions : list of OdeResult
        New list with noisy .y values (all else unchanged).
    """
    import copy

    if seed is not None:
        np.random.seed(seed)

    noisy_solutions = []

    for sol in solutions:
        noisy_sol = copy.deepcopy(sol)
        signal = sol.y
        noise = np.random.normal(loc=0.0, scale=noise_std, size=signal.shape)
        noisy_sol.y = signal + noise
        noisy_solutions.append(noisy_sol)

    return noisy_solutions

def plot_clean_vs_noisy(solution_clean, solution_noisy, var_names, var_idx=1):
    """
    Plot clean and noisy trajectory for one variable.
    """
    t = solution_clean.t
    y_clean = solution_clean.y[var_idx]
    y_noisy = solution_noisy.y[var_idx]

    plt.figure(figsize=(7, 4))
    plt.plot(t, y_clean, label="Clean", color="tab:blue")
    plt.plot(t, y_noisy, label="Noisy", color="tab:red", linestyle="--")
    plt.title(f"Clean vs. Noisy Trajectory - {var_names[var_idx]}")
    plt.xlabel("Time")
    plt.ylabel(var_names[var_idx])
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_noise_difference(solution_clean, solution_noisy, var_names, var_idx=1):
    """
    Plot the difference between clean and noisy trajectory.
    """
    t = solution_clean.t
    diff = solution_noisy.y[var_idx] - solution_clean.y[var_idx]

    plt.figure(figsize=(7, 4))
    plt.plot(t, diff, color="darkorange")
    plt.title(f"Noise Difference (Noisy - Clean) - {var_names[var_idx]}")
    plt.xlabel("Time")
    plt.ylabel("Difference")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Use hydra to configure the dataset creation along with the setup_dataset.yaml file
@hydra.main(config_path="src/conf", config_name="setup_dataset.yaml",version_base=None)
def main(config):

    var_names = ["theta", "omega", "E_d_dash", "E_q_dash", "R_F", "V_r", "E_fd", "P_sv", "P_m"]


    print("Starting dataset comparison script...")
    print("CUDA available:", torch.cuda.is_available())

    # Create model instance
    SM_model = ODE_modelling(config)
    modelling_full = SynchronousMachineModels(config)

    # ----- Generate dataset using set3 -----
    print("\n--- Generating data with set3 (grid sampling) ---")
    init_conditions_set3 = SM_model.create_init_conditions_set3()
    solution_set3 = SM_model.solve_sm_model(init_conditions_set3, modelling_full, flag_time=True)

    # ----- Generate dataset using set4 -----
    print("\n--- Generating data with set4 (LHS/random sampling) ---")
    init_conditions_set4 = SM_model.create_init_conditions_set4()
    solution_set4 = SM_model.solve_sm_model(init_conditions_set4, modelling_full, flag_time=True)

    # --- Compute residuals for set4 (used for Evo) ---
    print("\n--- Computing residuals for set3 ---")
    residuals_set3 = [
        compute_ode_residual(sol, modelling_full.odequations)
        for sol in solution_set3
    ]
    print(f"Set3 - Mean Residual: {np.mean(residuals_set3):.2e}, Max: {np.max(residuals_set3):.2e}")

    print("\n--- Computing residuals for set4 ---")
    residuals_set4 = [
        compute_ode_residual(sol, modelling_full.odequations)
        for sol in solution_set4
    ]
    print(f"Set4 - Mean Residual: {np.mean(residuals_set4):.2e}, Max: {np.max(residuals_set4):.2e}")

    # --- Set 5: Evo Sampling based on set4 residuals ---
    print("\n--- Generating data with set5 (Evo Sampling) ---")
    init_conditions_set5 = SM_model.create_init_conditions_set5(
        previous_ICs=init_conditions_set4,
        previous_errors=residuals_set4,
        total_samples=1000,
        exploration_ratio=0.2
    )
    solution_set5 = SM_model.solve_sm_model(init_conditions_set5, modelling_full, flag_time=True)
    noisy_solution_set5 = add_noise_to_solutions(solution_set5, noise_std=0.01, seed=42)

    # --- Compute residuals for Evo ICs ---
    residuals_set5 = [
        compute_ode_residual(sol, modelling_full.odequations)
        for sol in solution_set5
    ]
    residuals_set5_noisy = [
        compute_ode_residual(sol, modelling_full.odequations)
        for sol in noisy_solution_set5
    ]
    print(f"Set5 - Mean Residual: {np.mean(residuals_set5):.2e}, Max: {np.max(residuals_set5):.2e}")
    print(f"Set5 Noisy - Mean Residual: {np.mean(residuals_set5_noisy):.2e}, Max: {np.max(residuals_set5_noisy):.2e}")
    increase = np.mean(residuals_set5_noisy) - np.mean(residuals_set5)
    print(f"→ Δ Mean Residual due to noise: {increase:.2e}")
    # Example for one trajectory
    signal_std = np.std(solution_set5[0].y)
    noise_std = 0.01
    print(f"Noise-to-signal ratio: {noise_std / signal_std:.2%}")

    plot_clean_vs_noisy(solution_set5[0], noisy_solution_set5[0], var_names, var_idx=1)  # omega
    plot_noise_difference(solution_set5[0], noisy_solution_set5[0], var_names, var_idx=1)

    check_high_error_focus(init_conditions_set4, residuals_set4, init_conditions_set5, top_k=900)
    plot_exploit_explore_distribution(
        exploit_ics=init_conditions_set5[:800],
        explore_ics=init_conditions_set5[800:],
        var_names=["theta", "omega"]
    )

    # ----- Plot one example trajectory from each set (subplots) -----    
    t_vector = solution_set3[0].t  # use t from any solution

    plot_trajectories_with_derivatives(solution_set3, solution_set4, solution_set5, var_names)

    activity_set3 = compute_activity_for_all_solutions(solution_set3, var_names)
    activity_set4 = compute_activity_for_all_solutions(solution_set4, var_names)
    activity_set5 = compute_activity_for_all_solutions(solution_set5, var_names)


    plt.figure(figsize=(8, 5))
    plt.hist(activity_set3, bins=30, alpha=0.5, label='Set 3 (Grid)', color='royalblue', edgecolor='black')
    plt.hist(activity_set4, bins=30, alpha=0.5, label='Set 4 (LHS)', color='darkorange', edgecolor='black')
    plt.hist(activity_set5, bins=30, alpha=0.5, label='Set 5 (Evo)', color='forestgreen', edgecolor='black')

    plt.xlabel("Mean L2 Derivative (Activity)")
    plt.ylabel("Number of Trajectories")
    plt.title("Activity Distribution Comparison Across Sets")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


    # fig, axs = plt.subplots(3, len(var_names), figsize=(18, 9), sharex=True)

    # for i, var in enumerate(var_names):
    #     # Set 3
    #     axs[0, i].plot(t_vector, solution_set3[0].y[i])
    #     axs[0, i].set_title(var)
    #     axs[0, i].grid(True)

    #     # Set 4
    #     axs[1, i].plot(t_vector, solution_set4[0].y[i])
    #     axs[1, i].grid(True)

    #     # Set 5 (Evo)
    #     axs[2, i].plot(t_vector, solution_set5[0].y[i])
    #     axs[2, i].grid(True)

    # # Label rows
    # axs[0, 0].set_ylabel("Set 3")
    # axs[1, 0].set_ylabel("Set 4")
    # axs[2, 0].set_ylabel("Set 5 (Evo)")

    # fig.suptitle("Comparison of First Trajectory Across Sampling Strategies", fontsize=16)
    # plt.tight_layout()
    # plt.show()

    # --- IC Distribution Plots ---
    plot_ic_distributions_combined(
        ic_sets=[init_conditions_set3, init_conditions_set4, init_conditions_set5],
        var_names=var_names,
        titles=["Set 3: Grid", "Set 4: Random/LHS", "Set 5: Evo"]
    )

    # --- Residual Histograms ---
    plot_residual_histograms_combined(
        residual_sets=[residuals_set3, residuals_set4, residuals_set5],
        labels=["Set 3", "Set 4", "Set 5 (Evo)"]
    )

    return None

if __name__ == "__main__":
    main()

