import os
from src.ode.sm_models_o_e import SynchronousMachineModels
from src.nn.nn_actions_e import NeuralNetworkActions
from src.functions import *
from omegaconf import OmegaConf
import wandb

import gc
gc.collect()
# torch.cuda.empty_cache()
# torch.cuda.empty_cache()
# gc.collect()

# Define parameter grids
seeds = [3]
nn_types = ["KAN"]
optimizers = ["LBFGS"]
update_weight_methods = ["Static"]
activation_functions = ["tanh"]#["tanh", "swish", "cos"]
weight_combinations = [[1, 1e-3, 1e-4, 1e-3]]#[[1e2, 0, 1, 1e1]]#[[1, 1e-3, 1e-4, 1e-3]]
actnet_freqs = [10]#[2, 10,18]
actnet_hidden_dims = [25]#[15, 25,50]
actnet_hidden_layers = [2]#[1,2]
actnet_freq_scalings = [True]
actnet_freq_scaling_epsilons = [1e-2]#[1e-2, 1e-3]
kan_G_values = [7]#[5, 7, 10]
kan_k_values = [3]
kan_layer_values = [1]
kan_hidden_dims = [22]#[15, 22, 29]
skip_traj=[23]#[23,46,69]
skip_points=[38]#[19,38,57]

for seed in seeds:
    for nn_type in nn_types:
        for optimizer in optimizers:
            for update_method in update_weight_methods:
                for activation in activation_functions:
                    for weights in weight_combinations:
                            for s_t in skip_traj:
                                for s_p in skip_points:
                                    cfg = OmegaConf.load("src/conf/setup_dataset_nn_o_e.yaml")
                                    cfg.seed = seed
                                    cfg.nn.type = nn_type
                                    cfg.nn.weighting.flag_mean_weights = True
                                    cfg.nn.weighting.update_weight_method = update_method
                                    cfg.nn.weighting.weights = weights
                                    cfg.nn.activation_function = activation

                                    if nn_type == "actnet":
                                        for freq in actnet_freqs:
                                            for hidden_dim in actnet_hidden_dims:
                                                for hidden_layers in actnet_hidden_layers:
                                                    for freq_scaling in actnet_freq_scalings:
                                                        for freq_scaling_eps in actnet_freq_scaling_epsilons:
                                                            print(freq,hidden_dim,hidden_layers,freq_scaling,freq_scaling_eps)
                                                            cfg.nn.num_freqs = freq
                                                            cfg.nn.hidden_dim = hidden_dim
                                                            cfg.nn.hidden_layers = hidden_layers
                                                            cfg.nn.freq_scaling = freq_scaling
                                                            cfg.nn.freq_scaling_eps = freq_scaling_eps
                                                            lbfgs_iter = 10
                                                            cfg.nn.early_stopping_patience = int(cfg.nn.early_stopping_patience / lbfgs_iter)
                                                            cfg.nn.num_epochs = 1000
                                                            cfg.nn.weighting.update_weights_freq *= 4
                                                            cfg.nn.num_epochs = 1000 #int(cfg.nn.num_epochs / lbfgs_iter)

                                                            run = wandb.init(project=cfg.wandb.project)

                                                            # num_of_skip_data_points = 23
                                                            # num_of_skip_col_points = 19
                                                            cfg.nn.num_of_skip_data_points = s_t
                                                            cfg.nn.num_of_skip_col_points = s_p
                                                            num_of_skip_data_points = s_t
                                                            num_of_skip_col_points = s_p
                                                            num_of_skip_val_points = 4
                                                            # cfg.nn.batch_size=32
                                                            modelling_full = SynchronousMachineModels(cfg)
                                                            network2 = NeuralNetworkActions(cfg, modelling_full)

                                                            network2.pinn_train2(num_of_skip_data_points, num_of_skip_col_points, num_of_skip_val_points, run)

                                                            run.finish()
                                    elif nn_type == "KAN":
                                        for G in kan_G_values:
                                            for k in kan_k_values:
                                                for layers in kan_layer_values:
                                                    for hidden_dim in kan_hidden_dims:
                                                        cfg.nn.G = G
                                                        cfg.nn.k = k
                                                        cfg.nn.layers = layers
                                                        cfg.nn.hidden_dim = hidden_dim
                                                        cfg.nn.hidden_layers = 2
                                                        cfg.nn.hidden_dim = 12
                                                        cfg.nn.num_epochs = int(cfg.nn.num_epochs)
                                                        lbfgs_iter = 10
                                                        cfg.nn.early_stopping_patience = int(cfg.nn.early_stopping_patience / lbfgs_iter)
                                                        cfg.nn.num_epochs = 1000
                                                        cfg.nn.weighting.update_weights_freq *= 4
                                                        cfg.nn.num_epochs = 1000 #int(cfg.nn.num_epochs / lbfgs_iter)

                                                        run = wandb.init(project=cfg.wandb.project)

                                                        # num_of_skip_data_points = 23
                                                        # num_of_skip_col_points = 19
                                                        cfg.nn.num_of_skip_data_points = s_t
                                                        cfg.nn.num_of_skip_col_points = s_p
                                                        num_of_skip_data_points = s_t
                                                        num_of_skip_col_points = s_p
                                                        num_of_skip_val_points = 4
                                                        # cfg.nn.batch_size=32
                                                        modelling_full = SynchronousMachineModels(cfg)
                                                        network2 = NeuralNetworkActions(cfg, modelling_full)

                                                        network2.pinn_train2(num_of_skip_data_points, num_of_skip_col_points, num_of_skip_val_points, run)

                                                        run.finish()
                                    else:
                                        # if optimizer == "LBFGS":
                                        lbfgs_iter = 10
                                        cfg.nn.early_stopping_patience = int(cfg.nn.early_stopping_patience / lbfgs_iter)
                                        cfg.nn.num_epochs = 1000
                                        cfg.nn.weighting.update_weights_freq *= 4
                                        cfg.nn.num_epochs = 1000 #int(cfg.nn.num_epochs / lbfgs_iter)

                                        run = wandb.init(project=cfg.wandb.project)

                                        # num_of_skip_data_points = 23
                                        # num_of_skip_col_points = 19
                                        cfg.nn.num_of_skip_data_points = s_t
                                        cfg.nn.num_of_skip_col_points = s_p
                                        num_of_skip_data_points = s_t
                                        num_of_skip_col_points = s_p
                                        num_of_skip_val_points = 4
                                        # cfg.nn.batch_size=32
                                        modelling_full = SynchronousMachineModels(cfg)
                                        network2 = NeuralNetworkActions(cfg, modelling_full)

                                        network2.pinn_train2(num_of_skip_data_points, num_of_skip_col_points, num_of_skip_val_points, run)

                                        run.finish()
