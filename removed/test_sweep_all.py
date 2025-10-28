import os
from src.ode.sm_models_d import SynchronousMachineModels
from src.nn.nn_actions import NeuralNetworkActions
from src.functions import *
from omegaconf import OmegaConf
import wandb

def train(config=None):
    run = wandb.init(config=config)
    config = run.config

    # Load base configuration from YAML
    cfg = OmegaConf.load("src/conf/setup_dataset_nn.yaml")
    cfg.seed = config.seed
    cfg.model.model_flag = config.model_flag
    cfg.nn.type = config.nn_type
    cfg.nn.weighting.flag_mean_weights = True
    cfg.nn.weighting.update_weight_method = config.update_weight_method
    cfg.nn.weighting.weights = [config.weight_data, config.weight_dt, config.weight_pinn, config.weight_pinn_ic]
    cfg.nn.batch_size = config.batch_size

    # Adjust settings based on nn_type and optimizer
    
    if cfg.nn.type == "KAN":
        cfg.nn.hidden_layers = 2
        cfg.nn.hidden_dim = config.hidden_dim
        cfg.nn.num_epochs = int(cfg.nn.num_epochs / 4)

    if cfg.nn.optimizer == "LBFGS":
        lbfgs_iter = 10
        cfg.nn.early_stopping_patience = int(cfg.nn.early_stopping_patience / lbfgs_iter)
        cfg.nn.num_epochs = int(cfg.nn.num_epochs / lbfgs_iter)
        cfg.nn.weighting.update_weights_freq = int(cfg.nn.weighting.update_weights_freq*2) # increase due to internal iterations, around 25 internal iterations per epoch


    # Initialize model and network
    modelling_full = SynchronousMachineModels(cfg)
    network2 = NeuralNetworkActions(cfg, modelling_full)

    # Set skip points and start training
    num_of_skip_data_points = 23
    num_of_skip_col_points = 19
    num_of_skip_val_points = 4

    network2.pinn_train2(num_of_skip_data_points, num_of_skip_col_points, num_of_skip_val_points, run)
    run.finish()

if __name__ == "__main__":
    # Define sweep configuration
    sweep_config = {
        "method": "grid",
        "metric": {
            "name": "Test_loss",
            "goal": "minimize"
        },
        "parameters": {
            "seed": {"values": [1,3,7]},
            "nn_type": {"values": [ "DynamicNN"]},
            "batch_size" : {"values": [1024,None]},
            #"flag_mean_weights": {"values": [True, False]},
            "update_weight_method": {"values": ["Static"]},
            "weight_data": {"values": [1]},
            "weight_dt": {"values": [1e-3]},
            "weight_pinn": {"values": [1e-4]},
            "weight_pinn_ic": {"values": [1e-3]}
        }
    }
    model_flag_for_test = ["SM_AVR_GOV"]
    for model_flag in model_flag_for_test:
        sweep_config["parameters"]["model_flag"] = {"values": [model_flag]}
        # Initialize and run sweep
        sweep_id = wandb.sweep(sweep_config, project="Batching")
        wandb.agent(sweep_id, function=train)
