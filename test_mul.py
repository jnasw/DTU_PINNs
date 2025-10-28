import os
from src.ode.sm_models_o_e import SynchronousMachineModels
from src.nn.nn_dataset_pre import Datapreprocessor
from src.nn.nn_actions_pre import NeuralNetworkActions
from src.functions import *
from omegaconf import OmegaConf
import wandb
os.environ["WANDB_SILENT"] = "true"

comb = [
        [1, 1e-3, 1e-4, 1e-3]
        ]
seed = [1]
list = [ "Static"]
flag_mean_weights = [True]
nn_type = ["DynamicNN"]
for i in range(len(comb)):
    for j in range(len(seed)):
        for k in range(len(list)):
            for l in range(len(nn_type)):
                for r in range(len(flag_mean_weights)):

                    cfg = OmegaConf.load("src/conf/setup_dataset_nn.yaml")
                    cfg.seed = seed[j]
                    cfg.nn.type = nn_type[l]
                    cfg.nn.weighting.flag_mean_weights = flag_mean_weights[r]
                    cfg.nn.weighting.update_weight_method = list[k]
                    cfg.nn.weighting.weights = comb[i]

                    if cfg.nn.type == "KAN":
                        cfg.nn.hidden_layers = 2
                        cfg.nn.hidden_dim = 12 
                        cfg.nn.num_epochs = int(cfg.nn.num_epochs/5) #15000 for normal nn, and 3000 for kan. If LBFGS ->reduce to 1/10

                    if cfg.nn.optimizer == "LBFGS":
                        lbfgs_iter=10
                        cfg.nn.early_stopping_patience = int(cfg.nn.early_stopping_patience/lbfgs_iter)
                        cfg.nn.num_epochs = int(cfg.nn.num_epochs/lbfgs_iter) # reduce due to iterative internal epochs
                        cfg.nn.weighting.update_weights_freq = int(cfg.nn.weighting.update_weights_freq*4) # increase due to internal iterations

                    run = wandb.init(project=cfg.wandb.project)
                    num_of_skip_data_points = 23
                    num_of_skip_col_points = 19
                    num_of_skip_val_points = 4

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