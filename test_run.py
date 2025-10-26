# ==============================================================
# sanity test: Train PINN (data-only) on one dataset
# ==============================================================
import os
import torch
import pickle
import numpy as np
from omegaconf import OmegaConf
from src.nn.nn_actions import NeuralNetworkActions
from src.ode.sm_models_d import SynchronousMachineModels
from src.nn.nn_dataset import DataSampler
import torch.optim as optim


def load_dataset(dataset_path, device):
    with open(dataset_path, "rb") as f:
        dataset = pickle.load(f)

    x_all, y_all = [], []
    for r in dataset:
        t = np.array(r[0])
        y = np.vstack(r[1:]).T
        x = np.hstack([t.reshape(-1, 1), y])
        x_all.append(x)
        y_all.append(y)

    x_all = np.vstack(x_all)
    y_all = np.vstack(y_all)
    x_tensor = torch.tensor(x_all, dtype=torch.float32, device=device, requires_grad=True)
    y_tensor = torch.tensor(y_all, dtype=torch.float32, device=device)
    return x_tensor, y_tensor


def train_single_pinn(cfg, dataset_name="set5_mixed"):
    dataset_path = f"data/SM_AVR_GOV/dataset_{dataset_name}.pkl"
    model_save_path = f"model/SM_AVR_GOV/pinn_{dataset_name}_dataonly_test.pth"

    print(f"\n🚀 Training PINN on dataset: {dataset_path}")

    # --- Load dataset first ---
    ds = DataSampler(cfg, dataset_path=dataset_path)

    # --- Initialize model + network with preloaded data loader ---
    modelling_full = SynchronousMachineModels(cfg)
    network = NeuralNetworkActions(cfg, modelling_full, data_loader=ds)

    # Force data-only loss
    cfg.nn.weighting.weights = [1.0, 0.0, 0.0, 0.0]
    network.weight_data, network.weight_dt, network.weight_pinn, network.weight_pinn_ic = 1.0, 0.0, 0.0, 0.0

    # --- Split data ---
    x_train, y_train, x_col, x_ic, y_ic, x_val, y_val = ds.define_train_val_data2(
        cfg.dataset.perc_of_data_points,
        cfg.dataset.perc_of_col_points,
        1, 1, 1
    )

    device = network.device
    x_train = x_train.to(device).clone().detach().requires_grad_(True)
    y_train = y_train.to(device)
    x_val = x_val.to(device).clone().detach().requires_grad_(True)
    y_val = y_val.to(device)

    # --- Optimizer ---
    network.optimizer = optim.LBFGS(
        network.model.parameters(),
        lr=cfg.nn.lr,
        line_search_fn="strong_wolfe"
    )

    # --- Training loop ---
    print(f"Training {cfg.nn.type} for {cfg.nn.num_epochs} epochs (data-only)...")
    val_losses = []

    for epoch in range(cfg.nn.num_epochs):
        network.model.train()

        def closure():
            network.optimizer.zero_grad(set_to_none=True)
            y_hat, _, _ = network.calculate_point_grad2(x_train.clone().detach().requires_grad_(True), y_train)
            loss = network.criterion(y_hat, y_train)
            loss.backward()
            return loss

        loss = network.optimizer.step(closure)

        if (epoch + 1) % 10 == 0:
            network.model.eval()
            with torch.no_grad():
                y_val_pred = network.forward_pass(x_val)
                val_loss = network.criterion(y_val_pred, y_val).item()
                val_losses.append(val_loss)
            print(f"Epoch {epoch+1}/{cfg.nn.num_epochs} | Train Loss: {loss.item():.3e} | Val Loss: {val_loss:.3e}")

    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    torch.save({
        "model_state_dict": network.model.state_dict(),
        "config": OmegaConf.to_container(cfg, resolve=True),
        "val_losses": val_losses,
    }, model_save_path)

    print(f"✅ Saved model to: {model_save_path}\n")
    print(f"📉 Final validation loss: {val_losses[-1]:.3e}")


if __name__ == "__main__":
    cfg = OmegaConf.load("src/conf/setup_dataset_nn.yaml")
    cfg.nn.num_epochs = 50
    cfg.nn.early_stopping = False
    cfg.nn.lr = 1e-3
    cfg.nn.optimizer = "LBFGS"
    train_single_pinn(cfg, dataset_name="set5_mixed")