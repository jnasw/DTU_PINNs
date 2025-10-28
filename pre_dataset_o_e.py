from src.functions import *
from removed.params import *
import torch
import wandb
import hydra
from src.nn.nn_dataset_pre import Datapreprocessor
import os

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
os.environ["WANDB_SILENT"] = "true"

#os.chdir("/teamspace/studios/this_studio/Physics-Informed-Neural-Networks-for-Synchronous-Machine-Models/") #
os.environ["HYDRA_FULL_ERROR"]="1"
# from contextlib import chdir

# %cd /teamspace/studios/this_studio/Physics-Informed-Neural-Networks-for-Synchronous-Machine-Models

# Load config file using hydra
# @hydra.main(config_path="../Pinn-Thesis/src/conf", config_name="setup_dataset.yaml",version_base=None)
# Load config file using hydra
# @hydra.main(config_path="../Pinn-Thesis/src/conf", config_name="setup_dataset.yaml",version_base=None)
#@hydra.main(config_path=".\src\conf", config_name="setup_dataset.yaml",version_base=None)
#def main(config):

@hydra.main(config_path="src/conf", config_name="setup_dataset_nn_o_e.yaml",version_base=None)
def main(config):

    # Initialize wandb
    run = wandb.init(project=config.wandb.project)
    datapreprocessor = Datapreprocessor(config)
    datapreprocessor.get_preprocess_save_data()
    datapreprocessor.create_save_col_data()
    datapreprocessor.update_info_file()
    return None

if __name__ == "__main__":
    main()

