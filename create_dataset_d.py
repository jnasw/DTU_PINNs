from src.functions import *
import torch
import wandb
import hydra
import os
import numpy as np
from src.dataset.create_dataset_functions import ODE_modelling
from src.ode.sm_models_d import SynchronousMachineModels

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["HYDRA_FULL_ERROR"] = "1"


def compute_ode_residual(solution, ode_func):
    """
    Compute the mean squared residual for a trajectory from solve_ivp.
    """
    t = solution.t
    x = solution.y.T
    dxdt_numeric = np.gradient(x, t, axis=0)

    residuals = []
    for i in range(len(t)):
        dxdt_model = ode_func(t[i], x[i])
        diff = dxdt_numeric[i] - dxdt_model
        residuals.append(np.sum(diff ** 2))

    return np.mean(residuals)


# --- Main dataset creation pipeline ---
@hydra.main(config_path="src/conf", config_name="setup_dataset.yaml", version_base=None)
def main(config):
    """
    Automated dataset generation script for the PowerPINN Evo dataset family.
    
    This script:
    1. Generates a baseline random/LHS dataset (Set 4)
    2. Computes ODE residuals for that dataset
    3. Uses Evo sampling (create_init_conditions_set5) to generate 
       multiple derived datasets with different properties:
         - exploit (focus on high-error ICs)
         - explore (random)
         - mixed (balanced)
         - wide (expanded IC ranges)
         - mutated (add Gaussian perturbations)
         - sparse (fewer ICs)
         - dense (many ICs)
    4. Solves the ODE model for each and saves datasets.
    """

    # --- Initialize wandb run (safe for local/offline too) ---
    run = wandb.init(project=config.wandb.project if "wandb" in config else "PowerPINN",
                     name="dataset_generation",
                     mode="online")   # or "disabled" if you don’t want uploads

    print("\nStarting automated Evo dataset generation...")
    print("CUDA available:", torch.cuda.is_available())
    print("mps available:", torch.backends.mps.is_available())

    # --- Initialize system models ---
    SM_model = ODE_modelling(config)
    modelling_full = SynchronousMachineModels(config)

    # ==========================================================
    # STEP 1 — Generate base dataset (Set 4)
    # ==========================================================
    print("\n--- Generating base dataset: Set 4 (LHS/Random) ---")
    init_conditions_set4 = SM_model.create_init_conditions_set4(total_samples=1000)
    solution_set4 = SM_model.solve_sm_model(init_conditions_set4, modelling_full, flag_time=True)

    # --- Compute residuals for base dataset ---
    print("\n--- Computing residuals for Set 4 ---")
    residuals_set4 = [
        compute_ode_residual(sol, modelling_full.odequations)
        for sol in solution_set4
    ]
    print(f"Set4 residuals: mean={np.mean(residuals_set4):.2e}, max={np.max(residuals_set4):.2e}")

    # ==========================================================
    # STEP 2 — Define Evo dataset scenarios
    # ==========================================================
    evo_scenarios = [
        {
            "name": "set5_exploit",
            "params": dict(exploration_ratio=0.0, mutation_std=0.0, range_scale=1.0, total_samples=1000),
            "desc": "Pure exploitation — reuse only high-error ICs"
        },
        {
            "name": "set5_explore",
            "params": dict(exploration_ratio=1.0, mutation_std=0.0, range_scale=1.0, total_samples=1000),
            "desc": "Pure exploration — fully random new ICs"
        },
        {
            "name": "set5_mixed",
            "params": dict(exploration_ratio=0.2, mutation_std=0.0, range_scale=1.0, total_samples=1000),
            "desc": "Balanced exploit/explore (default)"
        },
        {
            "name": "set5_wide",
            "params": dict(exploration_ratio=0.2, mutation_std=0.0, range_scale=2.0, total_samples=1000),
            "desc": "Wider IC range (out-of-distribution)"
        },
        {
            "name": "set5_mutated",
            "params": dict(exploration_ratio=0.2, mutation_std=0.05, range_scale=1.0, total_samples=1000),
            "desc": "Add Gaussian noise to exploited ICs"
        },
        {
            "name": "set5_sparse",
            "params": dict(exploration_ratio=0.2, mutation_std=0.0, range_scale=1.0, total_samples=300),
            "desc": "Low-density dataset (fewer ICs)"
        },
        {
            "name": "set5_dense",
            "params": dict(exploration_ratio=0.2, mutation_std=0.0, range_scale=1.0, total_samples=2000),
            "desc": "High-density dataset (many ICs)"
        },
    ]

    # ==========================================================
    # STEP 3 — Generate Evo datasets
    # ==========================================================
    results_summary = []

    for scenario in evo_scenarios:
        name = scenario["name"]
        params = scenario["params"]
        desc = scenario["desc"]

        print(f"\n--- Generating {name} ---")
        print(f"Description: {desc}")
        print(f"Parameters: {params}")

        # Create initial conditions using Evo sampling
        init_conditions = SM_model.create_init_conditions_set5(
            previous_ICs=init_conditions_set4,
            previous_errors=residuals_set4,
            **params,
        )

        # Solve ODE model for new initial conditions
        solutions = SM_model.solve_sm_model(init_conditions, modelling_full, flag_time=True)

        # Compute ODE residual statistics
        residuals = [compute_ode_residual(sol, modelling_full.odequations) for sol in solutions]
        mean_resid = np.mean(residuals)
        max_resid = np.max(residuals)

        # Save dataset (assumes SM_model.save_dataset handles folder naming)
        SM_model.save_dataset(solutions, label=name)

        # Record summary
        results_summary.append({
            "name": name,
            "desc": desc,
            "samples": len(init_conditions),
            "mean_residual": mean_resid,
            "max_residual": max_resid
        })

        print(f"{name}: saved with {len(init_conditions)} ICs "
              f"| mean residual={mean_resid:.2e}, max={max_resid:.2e}")

    # ==========================================================
    # STEP 4 — Summary report
    # ==========================================================
    print("\nDataset generation complete. Summary:")
    print("----------------------------------------------------")
    for r in results_summary:
        print(f"{r['name']:>15} | N={r['samples']:>5} | mean={r['mean_residual']:.2e} | "
              f"max={r['max_residual']:.2e} | {r['desc']}")
    print("----------------------------------------------------")

    return None


if __name__ == "__main__":
    main()