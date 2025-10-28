import torch
import torch.nn as nn

from omegaconf import OmegaConf
import wandb
from src.nn.nn_model import Net, Network, PinnA, Kalm, FullyConnectedResNet, ActNet
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def find_files(cfg, dir, list_in_name=None):
    files = []
    for folder in dir:
        directory = os.path.join(cfg.dirs.model_dir,folder)
        for file in os.listdir(directory):
            if file.endswith(".pth"):
                file = os.path.join(folder, file)
                files.append(file)
      
    if list_in_name is not None:
        old_files = files
        files = []
        for file in old_files: # check if all the names are in the file
            cnt = 0
            for name in list_in_name:
                if name in file:
                    cnt += 1
            if cnt == len(list_in_name): # if all the names are in the file then append
                files.append(file)

    print("Files found", len(files))
    return files


def define_model_from_name(name):
    name_list = ["DynamicNN", "PinnA", "PinnAA", "PinnB", "KAN", "actnet"]
    #CHECK if name_list is in the name
    for n in name_list:
        if n in name:
            model_name = n
            return model_name   

def check_transform(name):
    transform_input_list = ["MinMax", "Std", "MinMax2"]
    for t1 in transform_input_list:
        if t1 in name:
            return True
        else:
            return False 
        
def check_sm_modelling(name, cfg):
    sm_modelling_list = ["SM_AVR_GOV", "SM_AVR", "SM_IB", "SM", "SM6"]
    for t1 in sm_modelling_list:
        if t1 in name:
            if t1 == cfg.model.model_flag:
                print(t1)
                return False
            else:
                return True
        else:
            return False

def define_nn_model(cfg, input_dim, output_dim):
    """
    This function defines the neural network model
    """
    print("Selected deep learning model: ",cfg.nn.type)        
    if cfg.nn.type == "KAN": # Static architecture of the neural network
        model = Kalm(input_dim, cfg.nn.hidden_dim, output_dim, cfg.nn.hidden_layers)
        # model.speed()
    elif cfg.nn.type == "StaticNN": # Static architecture of the neural network
        model = Net(input_dim, cfg.nn.hidden_dim, output_dim)
    elif cfg.nn.type == "actnet":
        model = ActNet(
                input_dim=input_dim,
                embed_dim= cfg.nn.hidden_dim,
                num_layers=cfg.nn.hidden_layers,
                out_dim=output_dim,
                num_freqs=cfg.nn.num_freqs,
                # w0_fixed=torch.pi,
                freq_scaling=cfg.nn.freq_scaling,
                freq_scaling_eps=cfg.nn.freq_scaling_eps,
            )
    elif cfg.nn.type == "DynamicNN" or cfg.nn.type == "PinnB" or cfg.nn.type == "PinnA": # Dynamic architecture of the neural network
        model = Network(input_dim, cfg.nn.hidden_dim, output_dim, cfg.nn.hidden_layers)
    elif cfg.nn.type == "PinnAA": # Dynamic architecture of the neural network with the PinnA architecture for the output
        model = PinnA(input_dim, cfg.nn.hidden_dim, output_dim, cfg.nn.hidden_layers)
    elif cfg.nn.type == "ResNet":
        num_blocks=2
        num_layers_per_block=2
        model = FullyConnectedResNet(input_dim, cfg.nn.hidden_dim, output_dim, num_blocks, num_layers_per_block)
    else:
        raise ValueError("Invalid nn type specified in the configuration.")
    return model
    


def forward_pass(model, data_network, input):
    """
    This function calculates the output of the neural network model, input is given as time and the other input columns
    """
    time = input[:,0].unsqueeze(1) # get the time column
    no_time = input[:,1:]
    model.eval()
    y_pred = model.forward(input)
    if data_network.cfg.nn.type == "PinnA":
        if data_network.cfg.dataset.transform_input == "None":
            return no_time + y_pred*time
        minus = data_network.data_loader.minus_input.clone().detach().to(data_network.device)
        divide = data_network.data_loader.divide_input.clone().detach().to(data_network.device)
        if data_network.cfg.dataset.transform_input == "MinMax2":
            div = nn.Parameter(torch.tensor(2.0), requires_grad=False)
            plus = nn.Parameter(torch.tensor(1.0), requires_grad=False)
            return ((no_time + plus) * divide[1:] / div + minus[1:]) + y_pred*((time + plus) * divide[0] / div + minus[0])
        return (no_time* divide[1:] + minus[1:]) + y_pred*(time* divide[0] + minus[0])
    if data_network.cfg.nn.type == "PinnB":
        return no_time + y_pred
    if data_network.cfg.nn.type == "DynamicNN" or data_network.cfg.nn.type == "PinnAA" or data_network.cfg.nn.type == "KAN" or data_network.cfg.nn.type == "actnet":
        return y_pred
    else:
        raise Exception('Enter valid NN type! (zeroth_order or first_order')
        
def load_model(model, cfg, name=None):
    """
    Load neural network model weights from the model_dir.

    Args:
        name (str): name of the model
    """
    # Load model from the model_dir
    model_dir = cfg.dirs.model_dir
    if not os.path.exists(model_dir) or len(os.listdir(model_dir)) == 0:
        raise Exception("No model found in the model_dir, please correct path and name of the model from the bucket first")
    if name is None:
        # Find first model in the model_dir
        name = os.listdir(model_dir)[0]
        if name == '.gitkeep':
            if len(os.listdir(model_dir)) == 1:
                raise Exception("No model found in the model_dir, please correct path and name of the model from the bucket first")
            name = os.listdir(model_dir)[1]
        print("Load model:", name)
    model_path = os.path.join(model_dir, name)
    if not os.path.exists(model_path):
        raise Exception("No model found in the model_dir, please correct path and name of the model from the bucket first")
    #check if torch load
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_data = torch.load(model_path, map_location=device)
    model.load_state_dict(model_data['model_state_dict'])
    return None

    
    

def data_input_target_limited(solution, time_limit):
    """
    Convert the loaded data into input and target data and limit the time to time_limit.

    Args:
        data (list): The loaded data.
    
    Returns:
        x_train_list (torch.Tensor): The input data.
        y_train_list (torch.Tensor): The target data.
    """
    dataset = []
    for i in range(len(solution)):
        r = [solution[i].t]  # append time to directory
        for j in range(len(solution[i].y)):
            r.append(solution[i].y[j])  # append the solution at each time step
        dataset.append(r)
    x_train_list = torch.tensor(())
    y_train_list = torch.tensor(())
    for training_sample in dataset:
        training_sample = torch.tensor(training_sample, dtype=torch.float32) # convert the trajectory to tensor
        y_train = training_sample[1:].T.clone().detach().requires_grad_(True) # target data
        training_sample_l = training_sample.T
        training_sample_l = training_sample_l[training_sample_l[:,0]<=time_limit].T # limit the time to time_limit
        y_train = training_sample_l[1:].T.clone().detach()
        x_train = training_sample_l.T
        x_train[:,1:]=x_train[0][1:]
        #discrard the first row of x_train and y_train as they are the same
        #x_train = x_train[1:]
        #y_train = y_train[1:]
        x_train = x_train.clone().detach().requires_grad_(True)
        x_train_list = torch.cat((x_train_list, x_train), 0)
        y_train_list = torch.cat((y_train_list, y_train), 0)
    return x_train_list, y_train_list

def forward_pass_b(model,x_train):
    """
    Perform a forward pass of the model.

    Args:
        model (torch.nn.Module): The neural network model.
        x_train (torch.Tensor): The input data.
    
    Returns:
        y_pred (torch.Tensor): The predicted target data.
    """
    y_pred = model(x_train)
    no_time = x_train[:,1:]
    return no_time+y_pred

def forward_pass_a(model,x_train):
    """
    Perform a forward pass of the model.

    Args:
        model (torch.nn.Module): The neural network model.
        x_train (torch.Tensor): The input data.
    
    Returns:
        y_pred (torch.Tensor): The predicted target data.
    """
    y_pred = model(x_train)
    time = x_train[:,0].unsqueeze(1) # get the time column
    no_time = x_train[:,1:]
    return no_time+time*y_pred


def predict(name, type, x_train_list,cfg):
    cfg.nn.type = type
    input_dim = x_train_list.shape[1]
    output_dim = x_train_list.shape[1]-1
    model = define_nn_model(cfg, input_dim, output_dim)
    load_model(model, cfg, name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    if type == "PinnA":
        y_pred = forward_pass_a(model,x_train_list)
    elif type == "PinnB":
        y_pred = forward_pass_b(model,x_train_list)
    else:
        y_pred = model.forward(x_train_list)
    return y_pred.detach().cpu().numpy()


def variable_names(cfg):
    modeling_guide_path = os.path.join(cfg.dirs.init_conditions_dir, "modellings_guide.yaml")
    modeling_guide = OmegaConf.load(modeling_guide_path)
    #check if proposed modeling is in the modeling guide
    for model in modeling_guide:
        model_name = model.get("name")
        if model_name == cfg.model.model_flag:
            keys = model.get("keys")
    return keys


# save as dat file
def save_as_dat(data, x_test, keys, file):
    
    time_t = x_test[:data.shape[0],0]
    time_t = time_t.cpu().detach().numpy()
    for i in range(x_test.shape[1]-1): # first column is time
        if i == 0:
            output_df = pd.DataFrame({
                't': time_t,
                keys[i]: data[:,i]
            })
        else:
            output_df[keys[i]] = data[:,i]
    if x_test.shape[1]==(data.shape[1]):
        output_df['total'] = data[:,i+1]
    file_name = file + ".dat"
    output_df.to_csv(file_name, sep=' ', index=False)
    return

def save_as_dat_per_var(data, x_test, num_of_points, keys, file):
    
    traj = int(data.shape[0]/num_of_points)

    data = data.reshape(traj, num_of_points, data.shape[1])#[50, 1000, 6]
    data = data.cpu().detach().numpy()
    time_t = x_test[:data.shape[1],0]
    time_t = time_t.cpu().detach().numpy()
    for i in range(data.shape[2]): # first column is time, and iterate over the variables
        for j in range(data.shape[0]):
            if j== 0:
                output_df = pd.DataFrame({
                    't': time_t,
                    str(j): data[j,:,i]
                })
            else:
                output_df[str(j)] = data[j,:,i]

        file_name = file + keys[i] + ".dat"
        output_df.to_csv(file_name, sep=' ', index=False)
    return

def loss_over_time(x_test, y_test, y_pred):

    unique_values = torch.unique(x_test[:,0]) # get the unique values of the time
    mae = []
    mse = []
    max_ae = []
    mean_mae = []
    mean_mse = []
    for value in unique_values: # for each time step
        index = torch.where(x_test[:,0] == value) # find the indexes of the time step
        y_pred_ = y_pred[index] # keep only the points at the specific time
        y_true = y_test[index] # keep only the points at the specific time
        mae_per_var = [] # calculate the mae for all the variables
        mse_per_var = [] # calculate the mse for all the variables
        max_ae_per_var = [] # calculate the max absolute error for all the variables
        for i in range(y_pred_.shape[1]):   # for each variable
            mae_per_var.append(torch.mean(torch.abs(y_pred_[:, i] - y_true[:, i])).item())
            mse_per_var.append(torch.mean((y_pred_[:, i] - y_true[:, i]) ** 2).item())
            max_ae_per_var.append(torch.max(torch.abs(y_pred_[:, i] - y_true[:, i])).item())
        
        mae_per_var.append(torch.mean(torch.abs(y_pred_ - y_true)).item())
        mse_per_var.append(torch.mean((y_pred_ - y_true) ** 2).item())
        max_ae_per_var.append(torch.max(torch.abs(y_pred_ - y_true)).item())
        mae.append((mae_per_var))
        mse.append((mse_per_var))
        max_ae.append((max_ae_per_var))
    
    mae = np.array(mae)
    mse = np.array(mse)
    max_ae = np.array(max_ae)
    return mae, mse, max_ae

def load_losses(files, x_test, cfg):
    for file in files:
        #remove the .pt from the file name
        file = file[:-4]
        new_file_path = "model/" + "paper/" + file
        file = "model/" + file
        keys = variable_names(cfg)
        print(keys)
        data_types = ["mae", "rmse"]
        for data_type in data_types:
            file_name = file + "_" + data_type + ".npy"
            data = np.load(file_name)
            print(data.shape)
            if data_type == "rmse":
                data_type = "mse"
            save_as_dat(data, x_test, keys, new_file_path + "_" + data_type)
    return

def find_min_loss(loss_list,name):
    min_loss = min(loss_list)
    min_index = loss_list.index(min_loss)
    min_name = name[min_index]
    return 

def sort_by_loss(loss_list,name):
    sorted_losses = sorted(loss_list)
    sorted_names = [x for _,x in sorted(zip(loss_list,name))]
    return sorted_losses, sorted_names


def loss_in_time(x_test, y_test, y_pred):
    unique_values = torch.unique(x_test[:,0]) # get the unique values of the time
    mae = []
    mse = []
    max_ae = []
    for value in unique_values: # for each time step
        index = torch.where(x_test[:,0] == value) # find the indexes of the time step
        # calculate the mae and rmse for each value
        y_pred_ = y_pred[index] # keep only the points at the specific time
        y_true = y_test[index] # keep only the points at the specific time
        mae.append(torch.mean(torch.abs(y_pred_ - y_true)).item())
        mse.append(torch.mean((y_pred_ - y_true)**2).item())
        max_ae.append(torch.max(torch.abs(y_pred_ - y_true)).item())
    #make them one numpy array
    metrics = [mae, mse, max_ae]
    metrics = np.array(metrics)

    
    return metrics

# save as dat file
def save_as_dat_in_time(data, x_test, keys, file):
    
    time_t = x_test[:data.shape[1],0]
    time_t = time_t.cpu().detach().numpy()
    for i in range(data.shape[0]): # first column is time
        if i == 0:
            output_df = pd.DataFrame({
                't': time_t,
                keys[i]: data[i,:]
            })
        else:
            output_df[keys[i]] = data[i,:]

    file_name = file + ".dat"
    output_df.to_csv(file_name, sep=' ', index=False)
    return

def plot_results(y_pred, y_test, x_test, num_of_points, cut_off=0):
    #create a plot that contains 4 subplots , 2 in each row
    plt.figure(figsize=(15, 15))
    #plt.suptitle("Comparison of the Prediction and the True Values for a Single Trajectory", fontsize=16)
    #put title for all the plots
    cut_off=0
    plt.subplot(4, 1, 1)
    #use time x_test[:,0] for the x-axis and the prediction and actual for y axi
    plt.plot(x_test[cut_off:num_of_points].cpu().detach().numpy()[:,0],y_pred.cpu().detach().numpy()[cut_off:num_of_points,0], label = "Prediction")
    plt.plot(x_test[cut_off:num_of_points].cpu().detach().numpy()[:,0],y_test[cut_off:num_of_points].cpu().detach().numpy()[:,0], label = "True")
    plt.xlabel("Time(s)")
    plt.ylabel("δ")
    #plt.legend()
    plt.subplot(4, 1, 2)
    plt.plot(x_test[cut_off:num_of_points].cpu().detach().numpy()[:,0],y_pred.cpu().detach().numpy()[cut_off:num_of_points,1], label = "Prediction")
    plt.plot(x_test[cut_off:num_of_points].cpu().detach().numpy()[:,0],y_test[cut_off:num_of_points].cpu().detach().numpy()[:,1], label = "True")
    plt.xlabel("Time(s)")
    plt.ylabel("ω (rad/s)")
    #plt.legend()
    plt.subplot(4, 1, 3)
    plt.plot(x_test[cut_off:num_of_points].cpu().detach().numpy()[:,0],y_pred.cpu().detach().numpy()[cut_off:num_of_points,2], label = "Prediction")
    plt.plot(x_test[cut_off:num_of_points].cpu().detach().numpy()[:,0],y_test[cut_off:num_of_points].cpu().detach().numpy()[:,2], label = "True")
    plt.xlabel("Time(s)")
    plt.ylabel("E_d_dash (pu)")
    #plt.legend()
    plt.subplot(4, 1, 4)
    plt.plot(x_test[cut_off:num_of_points].cpu().detach().numpy()[:,0],y_pred.cpu().detach().numpy()[cut_off:num_of_points,3], label = "Prediction")
    plt.plot(x_test[cut_off:num_of_points].cpu().detach().numpy()[:,0],y_test[cut_off:num_of_points].cpu().detach().numpy()[:,3], label = "True")
    plt.xlabel("Time(s)")
    plt.ylabel("E_q_dash (pu)")
    plt.legend()
    plt.show()



def save_results(model_flag, nn_type, values, values_one):
    file_name = "model\paper/" + "time_results0.txt"
    #check if file exists
    if not os.path.exists(file_name):
        with open(file_name, "w") as f:
            f.write("Model: " + model_flag + " NN: " + nn_type + " 50 trajectories Mean time: " + "{:.6f}".format(np.mean(values)) + " std time: " + "{:.6f}".format(np.std(values)) + " 1 trajectory Mean time: " + "{:.6f}".format(np.mean(values_one)) + " std time: " + "{:.6f}".format(np.std(values_one)) + "\n")
        return
    with open(file_name, "r") as f:
        content = f.read()
    with open(file_name, "a") as f:
        #check if already entry with model_flag and nn_type
        if any(model_flag in line and nn_type in line for line in content.splitlines()):
            print("Model and nn type already in file")
        else:
            f.write("Model: " + model_flag + " NN: " + nn_type + " 50 trajectories Mean time: " + "{:.6f}".format(np.mean(values)) + " std time: " + "{:.6f}".format(np.std(values)) + " 1 trajectory Mean time: " + "{:.6f}".format(np.mean(values_one)) + " std time: " + "{:.6f}".format(np.std(values_one)) + "\n")
    return