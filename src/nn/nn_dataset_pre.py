import pickle
import torch
import torch.nn as nn
import numpy as np
import os
import wandb
from omegaconf import OmegaConf
from pyDOE import lhs
from src.dataset.create_dataset_functions import ODE_modelling
from torch.utils.data import DataLoader, Dataset
import h5py
import re

    
class Datapreprocessor:
    """
    A class for preprocessing data for neural network training.

    Args:
        cfg (OmegaConf): The configuration for the dataset.

    Attributes:
        cfg (OmegaConf): The configuration for the dataset.
        device (torch.device): The device to use for training.
        model_flag (str): The type of modelling being used.
        seed (int): The seed for the random number generator.
        shuffle (bool): Whether to shuffle the data before splitting.
        time (float): The time limit for the data.
        num_of_points (int): The number of points in the data.
        input_dim (int): The dimension of the input data.
        total_trajectories (int): The total number of trajectories in the dataset.
        sample_per_traj (int): The number of samples per trajectory.
        num_samples (int): The number of total samples.
        split_ratio (float): The ratio of the training set over the validation and test sets.
        
        new_coll_points_flag (bool): Whether to use new collocation points.
        sampling (str): The sampling method for the initial conditions and possibly time.
        
        minus_input (torch.Tensor): The value that was subtracted from the input data.(if transformed)
        divide_input (torch.Tensor): The value that was divided from the above result.(if transformed)
        minus_target (torch.Tensor): The value that was subtracted from the target data.(if transformed)
        divide_target (torch.Tensor): The value that was divided from the above result.(if transformed)

    Methods:
        get_data: Load data from file and shuffle them.
        next_batch: Get the next batch of data.
        create_train_val_test_folder: Create the training, validation and testing folders
        data_input_target_limited: Convert the loaded data into input and target data with a time limit.
        train_val_test_split: Split the data into training, validation, and testing sets.
        find_norm_values: Find the min and max values of the data.
        find_std_values: Find the mean and std values of the data.
        define_minus_divide: Define the values to subtract and divide the data by.
        
        create_init_conditions_set: Create the initial conditions for the collocation points.
        create_col_points: Create the collocation points.
    """
    
    def __init__(self, cfg):
        self.cfg = cfg
        self.dataset_cfg = cfg.dataset
        self.model_flag = self.cfg.model.model_flag # 'SM_IB' or 'SM' or 'SM_AVR' or 'SM_GOV' or..
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.save_freq = 500000 # save every number of trajectories

        self.new_coll_points_flag = cfg.dataset.new_coll_points_flag # Define whether to use new collocation points
        
        self.seed = None if not hasattr(cfg.model, 'seed') else cfg.model.seed
        self.scrap_info()
        self.create_train_val_test_folder() # Create the training, validation and testing folders
        self.load_keys()
        """
        self.data, self.input_dim, self.total_trajectories = self.get_data() # Load the data from the file
        if self.input_dim != (self.ODE_modelling.keys_length + self.ODE_modelling.keys_ext_length + 1):
            raise Exception("The input dimension is not correct")
        self.output_dim = self.ODE_modelling.keys_length

        if self.new_coll_points_flag:
            self.sampling = cfg.model.sampling # Define the sampling method for the initial conditions and possibly time
        if self.cfg.dataset.transform_input != "None":
            self.minus_input, self.divide_input = self.define_minus_divide(self.x_train, self.x_train_col)
            self.minus_input = torch.nn.Parameter(self.minus_input, requires_grad=False)
            self.divide_input = torch.nn.Parameter(self.divide_input, requires_grad=False)
        if self.cfg.dataset.transform_output != "None":
            self.minus_target, self.divide_target = self.define_minus_divide(self.y_train, torch.empty(0))
            self.minus_target = torch.nn.Parameter(self.minus_target, requires_grad=False)
            self.divide_target = torch.nn.Parameter(self.divide_target, requires_grad=False)
        """
    def scrap_info(self):
        """
        This function scraps the info file and returns the values
        """
        number_of_dataset_folder = self.cfg.dataset.number # Define the number of the dataset to load
        self.folder_path = "./"+self.cfg.dirs.dataset_dir+"/" + self.model_flag + '/dataset_v' + str(number_of_dataset_folder)
        info_path = os.path.join(self.folder_path, "info.txt")
        self.info_path = info_path
        self.info_attributes = {}  # Dictionary to store dataset statistics
        if not os.path.exists(info_path):
            raise Exception("The info.txt file does not exist, please check the path")
        
        with open(info_path, "r") as text_file:
            lines = text_file.readlines()
            self.num_of_raw_files = int(lines[0].split(":")[1].strip())
            self.total_init_conditions = int(lines[2].split(":")[1].strip())
            self.time_sim = float(lines[3].split(":")[1].strip())
            self.num_of_points_sim = int(lines[4].split(":")[1].strip())
    
            # Define mappings for transformations
        trans_keys = {
            "min": "min",
            "max": "max",
            "range": "range",
            "mean": "mean",
            "std": "std",
            "n_samples": "n_samples"
        }
        for line in lines:
            parts = line.split(":")
            if len(parts) != 2:
                continue  # Skip malformed lines
            key, value = parts[0].strip().replace(" ", "_"), parts[1].strip()
            matched_key = next((k for k in trans_keys if k in key), None)
            if matched_key:
                
                if key == "n_samples":  # Integer value
                    setattr(self, trans_keys[matched_key], int(value))
                else:  # Numeric list (e.g., min, range, mean, std)
                    val = re.findall(r"[-+]?\d*\.\d+|\d+", value) 
                    tensor_values = [float(num) for num in val]
                    setattr(self, trans_keys[matched_key], torch.tensor(tensor_values))
        return

    def create_train_val_test_folder(self):

        
        
        if not os.path.exists(self.folder_path):
            raise Exception("The folder does not exist, please check the path")
        

        self.raw_data_path = self.folder_path + '/raw'
        if not os.path.exists(self.raw_data_path):
            raise Exception("The raw data folder does not exist, please check the path")
        
        
        # check that files in self.raw_data_path are equal to self.num_of_raw_files
        raw_files = os.listdir(self.raw_data_path)
        if len(raw_files) != self.num_of_raw_files:
            raise Exception("The number of raw files is not equal to the number of raw files in the info.txt file")

        # Create the training, validation and testing folders
        train_folder = self.folder_path + '/train'
        val_folder = self.folder_path + '/val'
        test_folder = self.folder_path + '/test'
        if not os.path.exists(train_folder):
            os.makedirs(train_folder)
        else:
            files = os.listdir(train_folder)
            #count how many have the "col" in their name
            files0 = [file for file in files if "col" in file]
            self.col_files = len(files0)
            print("Number of files in the train folder: ", len(files), "where ", len(files0), "are col files and", len(files)-len(files0), "are data files")
            
        if not os.path.exists(val_folder):
            os.makedirs(val_folder)
        else:
            files = os.listdir(val_folder)
            print("Number of files in the val folder: ", len(files))
        if not os.path.exists(test_folder):
            os.makedirs(test_folder)
        else:
            files = os.listdir(test_folder)
            print("Number of files in the test folder: ", len(files))

        self.train_folder = train_folder
        self.val_folder = val_folder
        self.test_folder = test_folder
        
        return
    
    def load_keys(self):
        modeling_guide_path = os.path.join(self.cfg.dirs.init_conditions_dir, "modellings_guide.yaml")
        if not os.path.exists(modeling_guide_path):
            raise ValueError(f"Modeling guide not found in {modeling_guide_path}")
        modeling_guide = OmegaConf.load(modeling_guide_path)
        for model in modeling_guide:
            model_name = model.get("name")
            if model_name == self.model_flag:
                keys = model.get("keys")
                #load keys_ext if exist in model
                keys_ext = model.get("keys_ext")
                #set attr length of keys and keys_ext
                keys_length = len(keys)
                keys_ext_length = len(keys_ext) if keys_ext else 0
                self.input_dim = keys_length + keys_ext_length + 1 # for time
                self.output_dim = keys_length
        if not keys:
            raise ValueError(f"Model {self.model_flag} not found in modeling guide")
        return 
    
    def set_info_attributes(self, **kwargs):
        """
        Store dataset statistics dynamically inside self.info_attributes.
        """
        self.info_attributes.update(kwargs)

    def update_info_file(self):
        """
        Update the info.txt file with values stored in self.info_attributes.
        """
        if not hasattr(self, "info_path") or not os.path.exists(self.info_path):
            raise Exception(f"Info file not found: {self.info_path}")

        with open(self.info_path, "r") as text_file:
            lines = text_file.readlines()

        # Overwrite lines 6 and onward with only relevant attributes
        lines[5:] = [f"{key.replace('_', ' ')}: {value}\n" for key, value in self.info_attributes.items()]

        with open(self.info_path, "w") as text_file:
            text_file.writelines(lines)

        print("Updated info.txt successfully.")
        return

    def get_preprocess_save_data(self):
        #iterate 
        print("Loading raw data from: ", self.raw_data_path)

        trajectories_cnt, train_traj_cnt, val_traj_cnt, test_traj_cnt = 0, 0, 0, 0
        train_cnt, val_cnt, test_cnt = 0, 0, 0

        x_train_chunk, y_train_chunk, x_val_chunk, y_val_chunk, x_test_chunk, y_test_chunk = torch.tensor(()), torch.tensor(()), torch.tensor(()), torch.tensor(()), torch.tensor(()), torch.tensor(())
        cnt = 0
        for file in os.listdir(self.raw_data_path):
            cnt += 1
            file_path = os.path.join(self.raw_data_path, file)
            sol = self.get_data(file_path, self.dataset_cfg.shuffle)
            if cnt == 1:
                self.sample_per_traj = len(sol[0][0])
            trajectories_cnt += len(sol)
            
            x_data, y_data = self.data_input_target_limited(sol, self.cfg.time)
            x_train, x_val, x_test, y_train, y_val, y_test = self.train_val_test_split(x_data, y_data, self.dataset_cfg.split_ratio, trajectories_cnt)
            train_traj_cnt += x_train.shape[0]
            val_traj_cnt += x_val.shape[0]
            test_traj_cnt += x_test.shape[0]
            
            x_train_chunk = torch.cat((x_train_chunk, x_train), 0)
            y_train_chunk = torch.cat((y_train_chunk, y_train), 0)
            x_val_chunk = torch.cat((x_val_chunk, x_val), 0)
            y_val_chunk = torch.cat((y_val_chunk, y_val), 0)
            x_test_chunk = torch.cat((x_test_chunk, x_test), 0)
            y_test_chunk = torch.cat((y_test_chunk, y_test), 0)

            if x_train_chunk.shape[0] > self.save_freq or cnt == self.num_of_raw_files:
                train_cnt += 1
                self.update_mean_std(x_train_chunk)
                self.update_min_max(x_train_chunk)
                self.save_dataset(x_train_chunk, y_train_chunk, self.train_folder, train_cnt)
                x_train_chunk, y_train_chunk = torch.tensor(()), torch.tensor(())

            if x_val_chunk.shape[0] > self.save_freq or cnt == self.num_of_raw_files:
                val_cnt += 1
                self.save_dataset(x_val_chunk, y_val_chunk, self.val_folder, val_cnt)
                x_val_chunk, y_val_chunk = torch.tensor(()), torch.tensor(())

            if x_test_chunk.shape[0] > self.save_freq or cnt == self.num_of_raw_files:
                test_cnt += 1
                self.save_dataset(x_test_chunk, y_test_chunk, self.test_folder, test_cnt)
                x_test_chunk, y_test_chunk = torch.tensor(()), torch.tensor(())

        # assign train_cnt etc to attributes 


        self.set_info_attributes(
            num_of_train_files=train_cnt,
            num_of_training_trajectories=train_traj_cnt,
            num_of_val_files=val_cnt,
            num_of_validation_trajectories=val_traj_cnt,
            num_of_test_files=test_cnt,
            num_of_testing_trajectories=test_traj_cnt
        )
        return 
    


    def create_save_col_data(self, save_flag = True):
        ODE_model = ODE_modelling(self.cfg)
        print("Generating collocation points")
        variables, set_of_values, iterations = ODE_model.create_init_conditions_info() # create the collocation points information
        init_condition_table = ODE_model.create_init_table2(set_of_values, iterations) # create the initial conditions table
        
        for i in range(0, len(init_condition_table), self.save_freq//self.cfg.num_of_points):
            x_train_col, x_train_col_zero = self.create_col_points(init_condition_table[i : (i + self.save_freq) ])  # Process chunk
            if save_flag:
                self.update_mean_std(x_train_col)
                self.update_min_max(x_train_col)
                self.save_dataset(x_train_col, None, self.train_folder, self.col_files + 1 + i // (self.save_freq//self.cfg.num_of_points), name0="_data_col")
                self.save_dataset(x_train_col_zero, x_train_col_zero[:,1:self.output_dim+1], self.train_folder, self.col_files +  1 + i // (self.save_freq//self.cfg.num_of_points), name0="_data_init") 
            else:
                return x_train_col, x_train_col_zero, x_train_col_zero[:,1:].detach() # this needs optimization, fix due to 
        self.set_info_attributes(
            num_of_train_col_files=(i+1)+self.col_files,
            num_of_training_col_trajectories=len(init_condition_table),
            )
        return 

    
    def create_col_points(self,init_condition_table):
        """
        Create the collocation points for the neural network training.

        Returns:
            torch.Tensor: The collocation points.
        """

        init_condition_table = torch.tensor(init_condition_table).float()
        col_points_zero_tensor = init_condition_table.clone().detach()
        time = torch.linspace(0, self.cfg.time, self.cfg.num_of_points + 1).unsqueeze(1)
        
        # Expand col_points to match time points efficiently
        init_condition_table = init_condition_table.unsqueeze(1).expand(-1, time.shape[0], -1)  # (N, T, D)
        time = time.expand(init_condition_table.shape[0], -1, 1)  # (N, T, 1)

        # Concatenate time with collocation points
        col_points_tensor = torch.cat((time, init_condition_table), dim=2)  # (N, T, D+1)
        
        return col_points_tensor.view(-1, col_points_tensor.shape[-1]), torch.cat((torch.zeros(col_points_zero_tensor.size(0), 1), col_points_zero_tensor), dim=1)

    def get_data(self, file_path, shuffle_flag):
        """
        Load and shuffle data from file.

        Returns:
            sol (list): The loaded data.
        """
        
        with open(file_path, 'rb') as f:
            sol = pickle.load(f)
        if shuffle_flag:
            np.random.shuffle(sol) # shuffle the trajectories as they are ordered
        return sol

    def data_input_target_limited(self, data, time_limit):
        """
        Convert the loaded data into input and target data and limit the time to time_limit.

        Args:
            data (list): The loaded data.
        
        Returns:
            x_train_list (torch.Tensor): The input data.
            y_train_list (torch.Tensor): The target data.
        """
        x_train_list, y_train_list = [], []
        if time_limit > self.time_sim:
            raise Exception("The time limit is greater than the simulation time")
        
        for training_sample in data:
            training_sample = torch.tensor(np.array(training_sample), dtype=torch.float32)  # Convert trajectory to tensor
            
            if time_limit != 0:
                training_sample = training_sample[:, training_sample[0] <= time_limit]  # Limit time to time_limit
            
            y_train = training_sample[1:self.output_dim+1].T.clone().detach()
            x_train = training_sample.T.clone().detach()
            x_train[:, 1:] = x_train[0, 1:]  # Replicate time column for all rows
            
            x_train_list.append(x_train)
            y_train_list.append(y_train)
    
        return torch.cat(x_train_list, dim=0), torch.cat(y_train_list, dim=0)
    
    def train_val_test_split(self, x_data, y_data, split_ratio, num_of_traj, val_flag=True):
        """
        Split the data into training, validation, and testing sets.
        
        Args:
            x_train (torch.Tensor): The input data.
            y_train (torch.Tensor): The target data.
            split_ratio (float): The ratio of the training set.

        Returns:
            x_train (torch.Tensor): The input data for the training set.
            x_val (torch.Tensor): The input data for the validation set.
            x_test (torch.Tensor): The input data for the testing set.
            y_train (torch.Tensor): The target data for the training set.
            y_val (torch.Tensor): The target data for the validation set.
            y_test (torch.Tensor): The target data for the testing set.
        """

        split = int(num_of_traj * split_ratio)*self.sample_per_traj # split trajectories and not just points into train, val and test
        if val_flag:
            val_split = int(num_of_traj * (10-split_ratio*10)/(2*10) )*self.sample_per_traj # multiply by 10 both denominator and numerator to get int values for sure
            x_train, x_val, x_test = x_data[:split], x_data[split:split+val_split], x_data[split+val_split:]
            y_train, y_val, y_test = y_data[:split], y_data[split:split+val_split], y_data[split+val_split:]
            return x_train, x_val, x_test, y_train, y_val, y_test
        else:
            x_train, x_test = x_data[:split], x_data[split:]
            y_train, y_test = y_data[:split], y_data[split:]
            return x_train, x_test, y_train, y_test

    def save_dataset(self, x_data, y_data, folder_path, cnt, name0 = "_data"):
        """
        Save the dataset to the folder path.
        """
        
        folders = ["train", "val", "test"]

        # find if keys in folders are in folder_path and assign the name that exists
        name = [folder for folder in folders if folder in folder_path][0]
        with h5py.File(os.path.join(folder_path, name + name0 + str(cnt)+'.h5'), 'w') as f:
            f.create_dataset('x_'+ name, data=x_data.cpu().numpy(), compression='gzip', compression_opts=9)
            if not name0=="_data_col":
                f.create_dataset('y_'+ name, data=y_data.cpu().numpy(), compression='gzip', compression_opts=9)

        print("Data saved to: ", os.path.join(folder_path, name + name0 + str(cnt)+'.h5'))
        return
    
    def load_dataset(self, folder_path, cnt):
        """
        Load the dataset from the folder path.
        """
        folders = ["train", "val", "test"]
        # find if keys in folders are in folder_path and assign the name that exists

        name = [folder for folder in folders if folder in folder_path][0]
        # find all files in the folder
        #files = os.listdir(folder_path)

        with h5py.File(os.path.join(folder_path, name + '_data'+str(cnt)+'.h5'), 'r') as f:
            x_data = torch.tensor(f['x_'+ name][:])
            y_data = torch.tensor(f['y_'+ name][:])
        return x_data, y_data



    def define_train_val_data2(self, perc_of_data, perc_of_col_data, num_of_skip_data_points, num_of_col_points, num_of_skip_val_points):
        """
        This function defines the training data
        Initially restrict the volume of data to both normal and collocation data
        Then sample from the data points and collocation points based on the given step size
        """
        perc_of_data = 1 if perc_of_data > 1 else perc_of_data # max percentage of data points
        perc_of_col_data = 1 if perc_of_col_data > 1 else perc_of_col_data # max percentage of collocation points
        num_of_data = int(perc_of_data * self.x_train.shape[0]) # max number of data points
        num_of_col_data = int(perc_of_col_data * self.x_train_col.shape[0]) # max number of collocation points
        x_train = self.x_train[:num_of_data: num_of_skip_data_points].clone().detach().to(self.device).requires_grad_(True) # training data after skipping points
        y_train = self.y_train[:num_of_data: num_of_skip_data_points].clone().detach().to(self.device)
        x_train_col = self.x_train_col[:num_of_col_data: num_of_col_points].clone().detach().to(self.device).requires_grad_(True) # traininng collaction points after skipping points
        x_train_col0 = self.x_train_col[self.x_train_col[:,0]==0].clone().detach().to(self.device).requires_grad_(True) # ic traininng collaction points 
        x_train_col0 = x_train_col0[:num_of_col_data].clone().detach().to(self.device).requires_grad_(True) # ic traininng collaction points after skipping points
        y_train_col0 = x_train_col0[:,1:self.output_dim+1].clone().detach().to(self.device) # ic training collocation points (when time is 0)
        x_val = self.x_val[:: num_of_skip_val_points].clone().detach().to(self.device).requires_grad_(True) # validation data without skipping points
        y_val = self.y_val[:: num_of_skip_val_points].clone().detach().to(self.device)
        self.training_shape, self.training_col_shape, self.training_col_shape0, self.validation_shape = (x_train.shape[0], x_train_col.shape[0], x_train_col0.shape[0], x_val.shape[0])
        if self.cfg.nn.type == "PinnA":
            y_train = y_train[x_train[:,0]!=0].clone().detach().to(self.device) # remove the 0 time  from the output data
            x_train = x_train[x_train[:,0]!=0].clone().detach().to(self.device).requires_grad_(True) # remove the 0 time  from the input data
            x_train_col = x_train_col[x_train_col[:,0]!=0].clone().detach().to(self.device).requires_grad_(True) # remove the 0 time from the input data
            #keep only the columns without time

        x_train = self.transform_input(x_train) # transform the input data according to the respective chosen input_transform method 
        x_train_col = self.transform_input(x_train_col) # transform the input data according to the respective chosen input_transform method
        x_val = self.transform_input(x_val) # transform the input data according to the respective chosen input_transform method
        x_train_col0 = self.transform_input(x_train_col0) # transform the input data according to the respective chosen input_transform method

        
        
        return x_train, y_train, x_train_col, x_train_col0, y_train_col0 , x_val, y_val
    
    def define_test_data(self,starting_traj,sample_per_traj,total_traj):

        folder_path = "./"+self.cfg.dirs.dataset_dir+"/" + self.cfg.model.model_flag + '/dataset_v' + str(self.cfg.dataset.number)
        test_dataset = HDF5DatasetStatic(folder_path, "test")
        x_test, y_test = test_dataset.x_data, test_dataset.y_data
        x_test = x_test[starting_traj*sample_per_traj:(starting_traj+total_traj)*sample_per_traj].clone().detach().to(self.device).requires_grad_(True)
        y_test = y_test[starting_traj*sample_per_traj:(starting_traj+total_traj)*sample_per_traj].clone().detach().to(self.device)
        if self.cfg.dataset.transform_input != "None":
            x_test = self.transform_input(x_test)
        return x_test, y_test

    
    def update_min_max(self, x_train_chunk):
        """
        Update the min and max values dynamically from different training dataset chunks.
        """
        if not hasattr(self, 'min') or not hasattr(self, 'max'):
            self.min = x_train_chunk.min(dim=0).values
            self.max = x_train_chunk.max(dim=0).values
        else:
            # Update min and max values
            self.min = torch.min(self.min, x_train_chunk.min(dim=0).values)
            self.max = torch.max(self.max, x_train_chunk.max(dim=0).values)

        # Ensure max-min is not zero by adding a small noise term
        self.range = self.max - self.min
        # Identify constant features (where range == 0)
        constant_mask = self.range == 0

        # Ensure range is not zero, but only for non-constant features
        self.range[~constant_mask] = self.range[~constant_mask].clamp(min=1e-7)

        # For constant features, set range to 1 to prevent division by zero
        self.range[constant_mask] = 1  # Prevent zero division

        self.set_info_attributes(
            min = self.min,
            max = self.max,
            range = self.range
        )
        return 
    
    def update_mean_std(self, training_chunk):
        """
        Update the mean and standard deviation dynamically from different training dataset chunks
        while handling varying subset sizes correctly.
        """
        batch_size = training_chunk.shape[0]

        if not hasattr(self, 'mean') or not hasattr(self, 'std') or not hasattr(self, 'n_samples'):
            self.mean = training_chunk.mean(dim=0)
            self.var = training_chunk.var(dim=0, unbiased=False)  # Store variance instead of std
            self.n_samples = batch_size
        #if self.var does not exist, them calculate it through self.std which exists
        if not hasattr(self, 'var'):
            self.var = self.std ** 2
        else:
            # Compute new mean and variance using Welford's method
            new_mean = training_chunk.mean(dim=0)
            new_var = training_chunk.var(dim=0, unbiased=False)  # Variance of the new batch

            n_total = self.n_samples + batch_size
            delta = new_mean - self.mean

            # Update mean
            self.mean += delta * (batch_size / n_total)

            # Update variance
            self.var = ((self.n_samples * self.var) + (batch_size * new_var) +
                        (self.n_samples * batch_size * (delta ** 2) / n_total)) / n_total

            self.n_samples = n_total  # Update total sample count

        # Compute standard deviation
        self.std = torch.sqrt(self.var)

        self.constant_features = self.std == 0

        # Ensure standard deviation is not zero, but only for non-constant features
        self.std[~self.constant_features] = self.std[~self.constant_features].clamp(min=1e-7)
        
        self.std[self.constant_features] = 1  # Prevent zero division

        self.set_info_attributes(
            mean = self.mean,
            std = self.std,
            n_samples = self.n_samples)
        
        return 
        


    
    def transform_data(self, data, flag):
        """
        This function transform the data 
        """
        if flag == "Input":
            data = (data - self.minus_input) / self.divide_input
        elif flag == "Input2":
            mul = nn.Parameter(torch.tensor(2.0).to(self.device), requires_grad=False)
            min = nn.Parameter(torch.tensor(1.0).to(self.device), requires_grad=False)
            data = mul * (data - self.minus_input) / self.divide_input - min
        else:
            raise Exception("Flag not implemented")
        return data

    def detransform_data(self, data, flag):
        """
        This function destandardize the data
        """  
        if flag == "Input":
            data = data * self.divide_input + self.minus_input
        elif flag == "Input2":
            mul = nn.Parameter(torch.tensor(2.0).to(self.device), requires_grad=False)
            min = nn.Parameter(torch.tensor(1.0).to(self.device), requires_grad=False)
            data = (data + min) * self.divide_input / mul + self.minus_input
        else:
            raise Exception("Flag not implemented")
        return data
    
    def transform_input(self, input):
        """
        This function transforms the input data
        """
        flag = "Input" # Flag to transform the input data
        if self.cfg.dataset.transform_input != "None":
            if self.cfg.dataset.transform_input == "Std":
                self.minus, self.divide = self.mean, self.std
                print("Standardizing input data for training")
            elif self.cfg.dataset.transform_input == "MinMax":
                self.minus, self.divide = self.min, self.range
                print("Normalizing input data for training")
            elif self.cfg.dataset.transform_input == "MinMax2":
                self.minus, self.divide = self.min, self.range
                print("Normalizing2 input data for training")
                flag = "Input2"
            else:
                raise Exception("Transformation not found")
            #input = self.transform_data(input,flag)
            #return input.clone().detach().to(self.device).requires_grad_(True)
            return self.transform_data(input,flag)
        else: 
            return input
        
    def detransform_input(self, input):
        """
        This function detransforms the input data
        """
        flag = "Input"
        if self.cfg.dataset.transform_input != "None":
            if self.cfg.dataset.transform_input == "Std":
                print("Unstandardizing input data ")
            elif self.cfg.dataset.transform_input == "MinMax":
                print("Unnormalizing input data")
            elif self.cfg.dataset.transform_input == "MinMax2":
                print("Unnormalizing input data")
                flag = "Input2"
            else:
                raise Exception("Transformation not found")
        #    input = self.detransform_data(input,flag)
        #return input.clone().detach().to(self.device).requires_grad_(True)
        return self.detransform_data(input,flag)








class HDF5Dataset(Dataset):
    def __init__(self, folder, data_type, col=False, shuffle=False):
        self.files = self.get_files(folder, data_type, col)
        self.shuffle = shuffle
        self.index_map = []  # Maps global indices to (file, row_idx)

        for file in self.files:
            with h5py.File(file, 'r') as h5f:
                num_samples = h5f['x_train'].shape[0]  # Number of rows
                self.index_map.extend([(file, i) for i in range(num_samples)])

        if self.shuffle:
            np.random.shuffle(self.index_map)

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        file, row_idx = self.index_map[idx]
        with h5py.File(file, 'r') as h5f:
            x_sample = h5f['x_train'][row_idx]  # Load only one row
            y_sample = h5f['y_train'][row_idx]

        return torch.tensor(x_sample, dtype=torch.float32), torch.tensor(y_sample, dtype=torch.float32)

    def get_files(self, folder, data_type, col=False):
        folder_path = os.path.join(folder, data_type)
        if data_type == "train":
            if col:
                #find only the files with col in their name
                files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(".h5") and "col" in f]
            else:
                files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(".h5") and "col" not in f]
        else:
            files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(".h5")]
        
        return files


class HDF5Dataset_static(Dataset):
    def __init__(self, folder, data_type, data_name=None, shuffle=False):
        self.files = self.get_files(folder, data_type, data_name)
        self.shuffle = shuffle
        if self.shuffle:
            np.random.shuffle(self.files)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file = self.files[idx]
        with h5py.File(file, 'r') as h5f:
            x_data = torch.tensor(h5f['x_train'][:], dtype=torch.float32)
            y_data = torch.tensor(h5f['y_train'][:], dtype=torch.float32)
        return x_data, y_data

    def get_files(self, folder, data_type, data_name):
        folder_path = os.path.join(folder, data_type)
        if data_type == "train":
            if data_name == "col":
                #find only the files with col in their name
                if data_name == "col_zero":
                    files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(".h5") and "col_zero" in f]
                else:
                    files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(".h5") and "col" in f]
            else:
                files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(".h5") and "col" not in f]
        else:
            files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(".h5")]
        print(files)
        return files
    

class HDF5DatasetStatic(Dataset):
    def __init__(self, folder, data_type, data_name=None, shuffle=False, step=1):

        self.flag_not_col = True #flag for collocation points
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.files = self.get_files(folder, data_type, data_name)
        self.shuffle = shuffle
        self.step = step
        self.x_data, self.y_data = self.load_all_data()
        

        if self.shuffle:
            indices = np.random.permutation(len(self.x_data))
            self.x_data = self.x_data[indices]
            if self.flag_not_col:
                self.y_data = self.y_data[indices]

    def load_all_data(self):
        x_list, y_list = [], []
        folders = ["train", "val", "test"]
        # find if keys in folders are in any of the file paths and assign the name that exists
        name = [folder for folder in folders if any(folder in file for file in self.files)][0]
        for file in self.files:
            with h5py.File(file, 'r') as h5f:
                x_list.append(torch.tensor(h5f['x_'+ name][::self.step], dtype=torch.float32))
                if self.flag_not_col:
                    y_list.append(torch.tensor(h5f['y_'+ name][::self.step], dtype=torch.float32))

        # Concatenate and flatten the batch dimension
        x_data = torch.cat(x_list, dim=0)  # Flatten (num_files * samples_per_file, features)
        if self.flag_not_col:
            y_data = torch.cat(y_list, dim=0)  # Flatten (num_files * samples_per_file, target_dim)
            if name in ["train", "val"]:
                return x_data.to(self.device), y_data.to(self.device)
            else:
                return x_data.to(self.device), y_data.to(self.device)
        else:
            return x_data.to(self.device), None



    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, idx):
        if self.flag_not_col:
            return self.x_data[idx], self.y_data[idx]
        else:
            return self.x_data[idx]

    def get_files(self, folder, data_type, data_name):
        folder_path = os.path.join(folder, data_type)
        if data_type == "train":
            if data_name == "col":
                self.flag_not_col = False
                files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(".h5") and "col" in f]
            elif data_name == "data_init":
                files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(".h5") and "data_init" in f]
            else:
                files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(".h5") and "col" not in f and "data_init" not in f]
        else:
            files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(".h5")]
        #print(files)
        return files
    
    