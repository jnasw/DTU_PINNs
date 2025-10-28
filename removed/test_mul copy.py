import os
from src.nn.nn_actions import NeuralNetworkActions
from src.functions import *
from omegaconf import OmegaConf
import wandb


comb = [
        [1, 1 , 1e-5 , 0],
        ]
seed = [1,3,7] 
list = ["Static","Dynamic"] 

for i in range(len(comb)):
    for j in range(len(seed)):
        for k in range(len(list)):
            
            cfg = OmegaConf.load("src/conf/setup_dataset_nn.yaml")
            cfg.nn.type = "DynamicNN"
            cfg.nn.optimizer = "LBFGS"
            cfg.nn.update_weight_method = list[k]
            if cfg.nn.optimizer == "LBFGS":
                lbfgs_iter=500
                cfg.nn.early_stopping_patience = int(cfg.nn.early_stopping_patience/lbfgs_iter)
                cfg.nn.num_epochs = int(cfg.nn.num_epochs/lbfgs_iter)

            run = wandb.init(project=cfg.wandb.project)

            print("Combination: ",i, " Seed: ",j, comb[i], seed[j])
            log_data_metrics_to_wandb(run,cfg)
            log_pinn_metrics_to_wandb(run,cfg)

            num_of_data = 4000000
            num_of_skip_data_points = 23
            num_of_skip_col_points = 5
            num_of_skip_val_points = 4

            perc_of_data = 1
            perc_of_pinn_data = 1


            cfg.seed = seed[j]
            network2 = NeuralNetworkActions(cfg)


            weight_data = comb[i][0]
            weight_dt = comb[i][1]
            weight_pinn =  comb[i][2]
            weight_pinn_ic = comb[i][3]


            network2.pinn_train( weight_data, weight_dt, weight_pinn, weight_pinn_ic, perc_of_data, perc_of_pinn_data, num_of_skip_data_points, num_of_skip_col_points, num_of_skip_val_points,run)


            run.finish()