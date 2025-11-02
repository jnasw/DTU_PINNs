# ==============================================================
# Train PINNs (data-only) across multiple datasets using LBFGS
# - Data-only gradient updates
# - Physics losses computed only for validation/logging
# ==============================================================

import os
import csv
import time
import torch
import numpy as np
from omegaconf import OmegaConf
import torch.optim as optim
import json

from src.nn.nn_dataset import DataSampler
from src.nn.nn_actions import NeuralNetworkActions
from src.ode.sm_models_d import SynchronousMachineModels


# -------------------------------------------------------------
# Utility: device detection
# -------------------------------------------------------------
def detect_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# -------------------------------------------------------------
# Training function
# -------------------------------------------------------------
def train_single_pinn_lbfgs(
    cfg,
    dataset_name,
    out_dir="model/SM_AVR_GOV",
    epochs=1000,
    lbfgs_lr=None,
    history_size=100,
    line_search_fn="strong_wolfe",
    tolerance_grad=1e-7,
    tolerance_change=1e-9,
    max_eval=None,
    print_every=50,
):
    """
    Train a PINN on a given dataset using LBFGS (data-only training).
    Physics-based losses (dt, pinn, pinn_ic) are computed only during validation
    and are logged to CSV and to the model checkpoint.
    """

    # -----------------------------
    # Setup & paths
    # -----------------------------
    dataset_path = f"data/SM_AVR_GOV/dataset_{dataset_name}.pkl"
    os.makedirs(out_dir, exist_ok=True)
    device = detect_device()

    print(f"\n🚀 Training PINN (LBFGS, data-only) on dataset: {dataset_path}")
    print(f"Device: {device}")

    # -----------------------------
    # Load data via framework (keeps shapes/transforms consistent)
    # -----------------------------
    ds = DataSampler(cfg, dataset_path=dataset_path)
    x_train, y_train, x_col, x_ic, y_ic, x_val, y_val = ds.define_train_val_data2(
        cfg.dataset.perc_of_data_points,
        cfg.dataset.perc_of_col_points,
        1, 1, 1
    )

    x_train, y_train = x_train.to(device), y_train.to(device)
    x_col, x_ic, y_ic = x_col.to(device), x_ic.to(device), y_ic.to(device)
    x_val, y_val = x_val.to(device), y_val.to(device)

    # -----------------------------
    # Initialize model/network
    # -----------------------------
    modelling_full = SynchronousMachineModels(cfg)
    network = NeuralNetworkActions(cfg, modelling_full, data_loader=ds)
    network.model.to(device)

    # -----------------------------
    # LBFGS optimizer (data-only)
    # -----------------------------
    if lbfgs_lr is None:
        lbfgs_lr = cfg.nn.lr

    network.optimizer = optim.LBFGS(
        network.model.parameters(),
        lr=lbfgs_lr,
        max_iter=20,                 # iterations per .step()
        max_eval=max_eval,
        tolerance_grad=tolerance_grad,
        tolerance_change=tolerance_change,
        history_size=history_size,
        line_search_fn=line_search_fn,
    )

    # -----------------------------
    # Logging setup
    # -----------------------------
    t0 = time.time()
    log_csv = os.path.join(out_dir, f"trainlog_{dataset_name}_LBFGS_e{epochs}.csv")
    with open(log_csv, "w", newline="") as f:
        csv.writer(f).writerow([
            "epoch", "train_loss", "val_data", "val_dt", "val_pinn", "val_pinn_ic",
            "val_total", "elapsed_s"
        ])

    metrics_history = []   # list of dicts (we’ll also store inside checkpoint)
    last_val_losses = {
        "data": float("nan"),
        "dt": float("nan"),
        "pinn": float("nan"),
        "pinn_ic": float("nan"),
        "total": float("nan"),
    }

    # -----------------------------
    # Validation helper (no grad)
    # -----------------------------
    def evaluate_val_losses():
        network.model.eval()
        # We need gradient tracking for autograd on the time column
        x_val_req = x_val.clone().detach().requires_grad_(True)
        x_col_req = x_col.clone().detach().requires_grad_(True)
        x_ic_req  = x_ic.clone().detach().requires_grad_(True)

        with torch.no_grad():
            # These are just for consistency; the derivative part needs requires_grad though
            pass

        # Compute validation components (gradient enabled for autograd)
        y_val_pred, dydt_val, ode_val = network.calculate_point_grad2(x_val_req, y_val)
        val_loss_data = network.criterion(y_val_pred, y_val)
        val_loss_dt = torch.mean(torch.stack([
            network.criterion(dydt_val[:, i], ode_val[:, i])
            for i in range(dydt_val.shape[1])
        ]))

        dydt_col, ode_col = network.calculate_point_grad2(x_col_req, None)
        val_loss_pinn = torch.mean(torch.stack([
            network.criterion(dydt_col[:, i], ode_col[:, i])
            for i in range(dydt_col.shape[1])
        ]))
        val_loss_ic = network.criterion(network.forward_pass(x_ic_req), y_ic)

        total_val = val_loss_data + val_loss_dt + val_loss_pinn + val_loss_ic

        return {
            "data": val_loss_data.item(),
            "dt": val_loss_dt.item(),
            "pinn": val_loss_pinn.item(),
            "pinn_ic": val_loss_ic.item(),
            "total": total_val.item(),
        }

    # -----------------------------
    # Training loop (data-only gradient)
    # -----------------------------
    print(f"LBFGS training for {epochs} epochs (data-only gradient updates)...")

    for epoch in range(1, epochs + 1):
        network.model.train()

        def closure():
            network.optimizer.zero_grad(set_to_none=True)
            y_hat = network.forward_pass(x_train)
            loss_data = network.criterion(y_hat, y_train)
            loss_data.backward()
            # stash for printing
            network._last_train_loss = loss_data.item()
            return loss_data

        network.optimizer.step(closure)
        train_loss = getattr(network, "_last_train_loss", 0.0)

        # Validation & logging (no gradients for physics)
        if epoch % print_every == 0 or epoch == 1 or epoch == epochs:
            val_losses = evaluate_val_losses()
            last_val_losses = val_losses  # keep most recent for checkpoint
            elapsed = time.time() - t0

            # Echo to console
            print(
                f"Epoch {epoch:4d}/{epochs} | "
                f"Train {train_loss:.3e} | "
                f"Val data {val_losses['data']:.3e} | "
                f"dt {val_losses['dt']:.3e} | "
                f"pinn {val_losses['pinn']:.3e} | "
                f"ic {val_losses['pinn_ic']:.3e} | "
                f"Total {val_losses['total']:.3e}"
            )

            # Append to CSV
            with open(log_csv, "a", newline="") as f:
                csv.writer(f).writerow([
                    epoch,
                    train_loss,
                    val_losses["data"],
                    val_losses["dt"],
                    val_losses["pinn"],
                    val_losses["pinn_ic"],
                    val_losses["total"],
                    f"{elapsed:.1f}",
                ])

            # Keep in-memory history (also saved into checkpoint)
            row = {
                "epoch": epoch,
                "train_loss": train_loss,
                **val_losses,
                "elapsed_s": elapsed,
            }
            metrics_history.append(row)

    # -----------------------------
    # Save model + metrics to disk
    # -----------------------------
    model_path = os.path.join(out_dir, f"pinn_{dataset_name}_LBFGS_e{epochs}.pth")
    torch.save(
        {
            "model_state_dict": network.model.state_dict(),
            "config": OmegaConf.to_container(cfg, resolve=True),
            "final_val_losses": last_val_losses,
            "metrics_history": metrics_history,
            "epochs": epochs,
            "optimizer": "LBFGS",
            "dataset_name": dataset_name,
        },
        model_path,
    )
    print(f"✅ Saved model: {model_path}")

    # Optional: write a one-line JSON summary for quick parsing later
    summary_json = os.path.join(out_dir, f"summary_{dataset_name}_LBFGS_e{epochs}.json")
    with open(summary_json, "w") as f:
        json.dump(
            {
                "dataset": dataset_name,
                "final": last_val_losses,
                "epochs": epochs,
                "checkpoint": os.path.basename(model_path),
                "log_csv": os.path.basename(log_csv),
            },
            f,
            indent=2,
        )
    print(f"📝 Wrote summary: {summary_json}")


# -------------------------------------------------------------
# Main script: loop over datasets
# -------------------------------------------------------------
if __name__ == "__main__":
    cfg = OmegaConf.load("src/conf/setup_dataset_nn.yaml")

    # Global training setup
    cfg.nn.early_stopping = False
    cfg.nn.type = "DynamicNN"
    cfg.nn.lr = 1e-3
    cfg.nn.weighting.update_weight_method = "Static"
    cfg.dataset.perc_of_data_points = 1
    cfg.dataset.perc_of_col_points = 1

    dataset_names = [
        "set5_exploit",
        "set5_explore",
        "set5_mixed",
        "set5_wide",
        "set5_mutated",
        "set5_sparse",
        "set5_dense",
    ]

    EPOCHS = 1000
    LBFGS_LR = 1e-3
    HIST_SIZE = 100
    PRINT_EVERY = 50

    for name in dataset_names:
        train_single_pinn_lbfgs(
            cfg,
            dataset_name=name,
            epochs=EPOCHS,
            lbfgs_lr=LBFGS_LR,
            history_size=HIST_SIZE,
            line_search_fn="strong_wolfe",
            tolerance_grad=1e-7,
            tolerance_change=1e-9,
            max_eval=None,
            print_every=PRINT_EVERY,
        )