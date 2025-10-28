import os
from src.ode.sm_models_o_e import SynchronousMachineModels
from src.nn.nn_actions_pre import NeuralNetworkActions
from src.nn.nn_dataset_pre import Datapreprocessor
from src.functions import *
from omegaconf import OmegaConf
import wandb

def train(config=None):
    run = wandb.init(config=config)
    config = run.config

    # Load base configuration from YAML
    cfg = OmegaConf.load("src/conf/setup_dataset_nn_o_e.yaml")
    cfg.seed = config.seed
    cfg.nn.type = config.nn_type
    cfg.nn.weighting.flag_mean_weights = True
    cfg.nn.weighting.update_weight_method = config.update_weight_method
    cfg.nn.weighting.weights = [config.weight_data, config.weight_dt, config.weight_pinn, config.weight_pinn_ic]

    if cfg.nn.type == "actnet":
        cfg.nn.num_freqs = config.actnet_freqs
        cfg.nn.hidden_dim = config.actnet_hidden_dim
        cfg.nn.hidden_layers = config.actnet_hidden_layers
        cfg.nn.freq_scaling = config.actnet_freq_scaling
        cfg.nn.freq_scaling_eps = config.actnet_freq_scaling_eps
    elif cfg.nn.type == "KAN":
        cfg.nn.G = config.kan_G
        cfg.nn.k = config.kan_k
        cfg.nn.layers = config.kan_layers
        cfg.nn.hidden_dim = config.kan_hidden_dim

    # Adjust settings based on nn_type and optimizer
    # if cfg.nn.type == "KAN":
    #     cfg.nn.hidden_layers = 2
    #     cfg.nn.hidden_dim = 12
    #     cfg.nn.num_epochs = int(cfg.nn.num_epochs / 5)

    # if cfg.nn.type == "actnet":
    #     .cfg.nn.num_freqs,
    #                 # w0_fixed=torch.pi,
    #                 freq_scaling=self.cfg.nn.freq_scaling,
    #                 freq_scaling_eps=self.cfg.nn.freq_scaling_eps,

    if cfg.nn.optimizer == "LBFGS":
        lbfgs_iter = 10
        cfg.nn.early_stopping_patience = int(cfg.nn.early_stopping_patience / lbfgs_iter)
        cfg.nn.num_epochs = 1000 #int(cfg.nn.num_epochs / lbfgs_iter)
        cfg.nn.weighting.update_weights_freq = int(cfg.nn.weighting.update_weights_freq*4) # increase due to internal iterations, around 25 internal iterations per epoch


    # Initialize model and network
    modelling_full = SynchronousMachineModels(cfg)
    datapreprocessor = Datapreprocessor(cfg)


    network = NeuralNetworkActions(cfg, modelling_full, datapreprocessor)

    # Set skip points and start training
    num_of_skip_data_points = 23
    num_of_skip_col_points = 19
    num_of_skip_val_points = 4
        
    # Train the network
    network.pinn_train(num_of_skip_data_points, num_of_skip_col_points, num_of_skip_val_points, run)
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
            "seed": {"values": [1, 3, 7]},
            "nn_type": {"values": [ "actnet","KAN","DynamicNN"]},
            "optimizer": {"values": ["LBFGS"]},
            #"flag_mean_weights": {"values": [True, False]},
            "update_weight_method": {"values": ["Static"]},
            "activation_function": {"values": ["tanh","swish","cos"]},
            "weight_data": {"values": [1]},
            "weight_dt": {"values": [1e-3]},
            "weight_pinn": {"values": [1e-4]},
            "weight_pinn_ic": {"values": [1e-3]},
            "actnet_freqs": {"values": [2, 8, 16]},
            "actnet_hidden_dim": {"values": [12, 24, 36]},
            "actnet_hidden_layers": {"values": [1, 2]},
            "actnet_freq_scaling":  {"values":[True, False]},
            "actnet_freq_scaling_eps":  {"values":[1e-2, 1e-3]},
            "kan_G":  {"values":[5,7,10]},
            "kan_k":  {"values":[3]},
            "kan_layers":  {"values":[1, 2]},
            "kan_hidden_dim":  {"values":[5,9,13]}
        }
    }
                    # num_freqs=self.cfg.nn.num_freqs,
                    # # w0_fixed=torch.pi,
                    # freq_scaling=self.cfg.nn.freq_scaling,
                    # freq_scaling_eps=self.cfg.nn.freq_scaling_eps,
    # Initialize and run sweep
    sweep_id = wandb.sweep(sweep_config, project="PINN-open-loop")
    wandb.agent(sweep_id, function=train)
