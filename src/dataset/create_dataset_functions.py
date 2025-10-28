from src.functions import *
from omegaconf import OmegaConf
import os
import wandb
import pickle
import numpy as np
from pyDOE import lhs
from scipy.integrate import solve_ivp
import time
import h5py

class ODE_modelling():
    def __init__(self, config):
        """
        Initializes an instance of ClassName.
        Args:
            config: The configuration object containing various parameters.

        Attributes:
            config (object): The configuration object.
            modelling_method (str): The modelling method: "Ground truth" or "Collocation".
            model_flag (str): The model flag: SM, SM_AVR or SM_AVR_GOV.
            time (str): The time interval for the simulation.
            machine_num (int): The machine number that will be used.
            init_conditions (str): The initial conditions set 
            params_dir (str): The directory for parameters: machine, avr, gov and system.
            init_conditions_dir (str): The directory for initial conditions per machine modelling type: SM, SM_AVR and SM_AVR_GOV.
            dataset_dir (str): The directory for saving the dataset.
            torch (bool): The flag to use PyTorch for the model.

        Methods:
            define_machine_params: Define the parameters of the synchronous machine based on the machine_num.
            define_system_params: Define the system parameters of the synchronous machine.
            append_element: Appends an element to each state in the given value set by iterating over a range of values.
            append_element_set: Appends an element to each state in the given value set by iterating over a range of values.
            create_init_conditions_set: Define the various initial conditions of the synchronous machine and return a matrix with all the possible combinations.
            create_solver: Create the solver for the synchronous machine model.
            solve_sm_model: Solves the synchronous machine model for multiple initial conditions.
            save_dataset: Create and save dataset for the model.
            load_dataset: Load the dataset.
        """
        self.config = config
        self.model_flag = config.model.model_flag
        self.time = config.time
        self.num_of_points = config.num_of_points
        self.init_condition_bounds = config.model.init_condition_bounds
        self.sampling = config.model.sampling
        self.torch = config.model.torch
        self.seed = None if not hasattr(config.model, 'seed') else config.model.seed
        self.load_keys()
    
    def append_element(self, value_set, Value_range, num_ranges):
        """
        Appends an element to each state in the given value set by iterating over a range of values.

        Args:
            value_set (list): The list of states to which the element will be appended.
            Value_range (tuple): The range of values from which the element will be selected.
            num_ranges (int): The number of ranges to divide the Value_range into.

        Returns:
            list: A new list of states with the element appended.

        """
        
        new_value_set = []
        for j in range(len(value_set)):
            for i in range(num_ranges):
                value = (Value_range[0] + i * (Value_range[1] - Value_range[0]) / (num_ranges - 1) if num_ranges > 1 else Value_range[0])
                new_state = value_set[j].copy()
                new_state.extend([value])
                new_value_set.append(new_state)
        return new_value_set

    def load_keys(self):

        init_conditions_dir = self.config.dirs.init_conditions_dir
        modeling_guide_path = os.path.join(init_conditions_dir, "modellings_guide.yaml")
        modeling_guide = OmegaConf.load(modeling_guide_path)
        for model in modeling_guide:
            model_name = model.get("name")
            if model_name == self.model_flag:
                keys = model.get("keys")
                self.fullname = model.get("fullname")
                #load keys_ext if exist in model
                keys_ext = model.get("keys_ext")
                #set attr length of keys and keys_ext
                self.keys_length = len(keys)
                self.keys_ext_length = len(keys_ext) if keys_ext else 0
                #append keys_ext to keys for the initial conditions check
                if keys_ext:
                    self.keys = keys + keys_ext
                else:
                    self.keys = keys
        if not keys:
            raise ValueError(f"Model {self.model_flag} not found in modeling guide")
        return 

    def check_ic_yaml(self,init_conditions):
        """
        Modeling guide contains all the variables that can be used in the modeling.
        Check if the variables in the initial conditions are in the modeling guide in the correct order
        """
        

        for i in range(len(init_conditions)):
            name = init_conditions[i].get("name")
            if name not in self.keys:
                raise ValueError(f"Variable {name} not found in modeling guide")
            if name != self.keys[i]: #same order as in the modeling guide
                raise ValueError(f"Variable {name} does not match the modeling variable {self.keys[i]}")
            #check if iterations are always a number:
            iterations = init_conditions[i].get("iterations")
            if not isinstance(iterations, int):
                raise ValueError(f"Variable {name} iterations must be an integer")
        return


    
    def append_element_set(self, value_set, Value_range, num_ranges):
        seed = (self.seed if self.seed is not None else np.random.randint(0, 1000))
        if self.sampling=="Random":
            points = np.random.uniform(0, 1, num_ranges)
            points = points.reshape(-1, 1)
        elif self.sampling=="Linear":
            points = np.linspace(0, 1, num_ranges)
            points = points.reshape(-1, 1)
        elif self.sampling=="Lhs":
            points = lhs(n=1, samples=num_ranges)
        else:
            raise Exception("Sampling method not implemented")
        
        new_value_set = []
        iterations = len(value_set) if len(value_set)>1 else 1
        for j in range(iterations):
            if len(value_set)<1:
                values = (Value_range[0] + points * (Value_range[1] - Value_range[0]) if num_ranges > 1 else Value_range[0])
                new_value_set = [values][0].tolist()
            else:
                for i in points:
                    if isinstance(i, np.ndarray):
                        i = i.item()
                    value = (Value_range[0] + i * (Value_range[1] - Value_range[0]) if num_ranges > 1 else Value_range[0])
                    new_state = value_set[j].copy()
                    if isinstance(new_state, np.ndarray):
                        new_state = new_state.tolist()
                    new_state.extend([value])
                    new_value_set.append(new_state)
        return new_value_set
    

    def create_init_table2(self, set_of_values, iterations):
        """
        Creates an initialization table for multiple variables.

        Parameters:
        - variables: List of dictionaries with keys:
        - "name": Variable name (not used in calculations but can be kept for reference).
        - "range": List containing either [min, max] or a single fixed value.
        - "iterations": Number of sample points for this variable.

        Returns:
        - A list of sampled initialization points, where each row is a combination of values for all variables.
        """

        sampled_points = []
        for i in range(len(set_of_values)):
            v_range = set_of_values[i]
            num_samples = iterations[i]

            if len(v_range) == 1 or num_samples == 1:
                # If only one value is needed, use it directly
                sampled_values = np.array([v_range[0]])  
            else:
                v_min, v_max = v_range  # Unpack range

                # Select sampling method
                if self.sampling == "Random":
                    points = np.random.uniform(0, 1, num_samples).reshape(-1, 1)
                elif self.sampling == "Linear":
                    points = np.linspace(0, 1, num_samples).reshape(-1, 1)
                elif self.sampling == "Lhs":
                    points = lhs(n=1, samples=num_samples)
                else:
                    raise Exception(f"Sampling method '{self.sampling}' not implemented.")

                # Scale to the desired range
                sampled_values = v_min + points * (v_max - v_min)

            sampled_points.append(sampled_values.flatten())

        # Create the initialization table by taking all possible combinations
        init_condition_table = np.array(np.meshgrid(*sampled_points)).T.reshape(-1, len(set_of_values))

        return init_condition_table.tolist()


    def create_init_conditions_info(self):
        """
        Define the various initial conditions of the synchronous machine and return a matrix with all the possible combinations.

        Returns:
            list: A matrix with all the possible combinations of initial conditions.
        """
        init_conditions_dir = self.config.dirs.init_conditions_dir
        if self.torch:# if using torch then use the nn_init_cond.yaml file to create collocation points init conditions
            init_conditions_path = os.path.join(init_conditions_dir, self.model_flag,"nn_init_cond"+str(self.init_condition_bounds)+".yaml")
        else: # 
            init_conditions_path = os.path.join(init_conditions_dir, self.model_flag,"init_cond"+str(self.init_condition_bounds)+".yaml")
        self.init_conditions_path = init_conditions_path
        init_conditions = OmegaConf.load(init_conditions_path)
        self.check_ic_yaml(init_conditions)

        for i in range(len(init_conditions)):
            if len(init_conditions[i]["range"]) == 1: # if unique value then set iterations to 1
                init_conditions[i]["iterations"] = 1 

        # Initialize the set_of_values and iterations lists and variables
        set_of_values = []
        iterations = []
        variables = []
        # Extract values, iterations and variables from init_conditions
        for condition in init_conditions:
            set_of_values.append(condition['range'])
            iterations.append(condition['iterations'])
            variables.append(condition['name'])

        # Calculate the number of different initial conditions
        number_of_conditions = 1
        for it in iterations:
            number_of_conditions *= it

        if self.torch:
            print("Number of different initial conditions for collocation points: ", number_of_conditions)
            #wandb.log({"Number of different initial conditions for collocation points: ": number_of_conditions})
        else:
            print("Number of different initial conditions: ", number_of_conditions)
            wandb.log({"Number of different initial conditions: ": number_of_conditions})

        print(variables, "Variables")
        print(set_of_values,"Set of values for init conditions")
        print(iterations,"Iterations per value")
        return variables, set_of_values, iterations
    
    def create_init_table(self, set_of_values, iterations):
        init_condition_table = []   
        for k in range(len(set_of_values)):
            init_condition_table = self.append_element_set(init_condition_table, set_of_values[k], iterations[k])
        return init_condition_table
    

    def create_init_conditions_set3(self):
        """
        Define the various initial conditions of the synchronous machine and return a matrix with all the possible combinations.
        """
        variables, set_of_values, iterations = self.create_init_conditions_info()
        init_condition_table = self.create_init_table(set_of_values, iterations)

        return init_condition_table

    def solve(self, x0, modelling_full, method=True): # method always must be true Modelling approach to be followed, ddelta = omega not ddelta = omega*Omega_B
        """
        Solve the differential equations for the synchronous machine model.

        Parameters:
        - x0: list of initial state variables

        Returns:
        - solution: solution of the differential equations
        """

        
        """
        # Initial state
        if self.model_flag=="SM_IB" or self.model_flag=="SM":
            x0 = [self.theta, self.omega, self.E_d_dash, self.E_q_dash]
        if self.model_flag=="SM_AVR":
            x0 = [self.theta, self.omega, self.E_d_dash, self.E_q_dash, self.R_F, self.V_r, self.E_fd]
        if self.model_flag=="SM_AVR_GOV":
            x0 = [self.theta, self.omega, self.E_d_dash, self.E_q_dash, self.R_F, self.V_r, self.E_fd, self.P_m, self.P_sv]
        """
        if method:
            solution = solve_ivp(modelling_full.odequations, self.t_span, x0, t_eval=self.t_eval)
        else:
            x0[1] = x0[1] / self.omega_B
            solution = solve_ivp(modelling_full.odequations_v2, self.t_span, x0, t_eval=self.t_eval)
        return solution

    def solve_sm_model(self, init_conditions, modelling_full, flag_time=False):
        """
        Solves the synchronous machine model for multiple initial conditions.

        Args:
            machine_params (dict): Dictionary containing the parameters of the synchronous machine.
            system_params (dict): Dictionary containing the parameters of the power system.
            init_conditions (list): List of initial conditions for which the model needs to be solved.

        Returns:
            list: List of solutions for each initial condition.

        """
        self.t_span, self.t_eval = set_time(self.time, self.num_of_points)
        self.total_init_conditions = len(init_conditions)
        self.save_flag = True
        solution_all=[]
        if flag_time:
            start_time = time.time()   
            start_per_iteration = start_time
            time_list = [] 
        for i in range(self.total_init_conditions): # iterate over all the initial conditions
            solution = self.solve(init_conditions[i], modelling_full) # solve the model for each initial condition
            solution_all.append(solution) # append the solution to the solution_all list

            if (i% self.config.save_freq == 0 and i>0) or i==(self.total_init_conditions-1): # save every save_freq iterations or when the last iteration is reached
                self.save_dataset(solution_all, i)
                solution_all = [] # reset the solution_all list

            if flag_time:
                end_per_iteration = time.time()
                time_list.append(end_per_iteration - start_per_iteration)
                start_per_iteration = end_per_iteration
            
        if flag_time:
            end_time = time.time()
            #print mean and std of time per iteration
            print("Mean time per iteration: ", np.mean(time_list), " and std: ", np.std(time_list))
            print(f"Time taken to solve the model for {self.total_init_conditions} initial conditions: {end_time - start_time} seconds.")
        return None
    
    
    
    def save_dataset(self, solution, iteration):
        """
        Create and save dataset for the model.

        Args:
            solution (list): The solution of the differential equations of the synchronous machine.

        Returns:
            list: The dataset of the synchronous machine.
        """
        dataset_dir = self.config.dirs.dataset_dir
        if self.save_flag: # create a new folder for the dataset only in the first iteration
            if not os.path.exists(os.path.join(dataset_dir, self.model_flag)):
                os.makedirs(os.path.join(dataset_dir, self.model_flag)) # create a new folder for the dataset if doesn't exist
            #find number of folders in the dataset directory
            self.num_of_folders = len([f for f in os.listdir(os.path.join(dataset_dir, self.model_flag)) if os.path.isdir(os.path.join(dataset_dir, self.model_flag, f))])
            self.dataset_folder_path = os.path.join(dataset_dir, self.model_flag, "dataset_v" + str(self.num_of_folders + 1),"raw")
            os.makedirs(self.dataset_folder_path)
            print(f"Created dataset folder: {os.path.dirname(self.dataset_folder_path)}")

            self.save_flag = False

        dataset_path = os.path.join(self.dataset_folder_path,"file" + str(iteration) + ".pkl")

        dataset = []
        for i in range(len(solution)):
            r = [solution[i].t]  # append time to directory
            # for j in range(self.keys_length): # save only the keys not the external keys
            for j in range(len(solution[i].y)): #must save all the keys due to the preprocessing of the NN dataset
                r.append(solution[i].y[j])  # append the solution at each time step
            dataset.append(r)

        # save the dataset as pickle in the dataset directory
        with open(dataset_path, 'wb') as f:
            pickle.dump(dataset, f)
        #with h5py.File(self.hdf5_path, 'w') as f:
        #    f.create_dataset('trajectories', data=np.array(dataset, dtype=np.float32), chunks=True, compression='gzip')
        # count the number of files in the directory
        if iteration == self.total_init_conditions - 1:
            num_of_files = len([f for f in os.listdir(self.dataset_folder_path) if os.path.isfile(os.path.join(self.dataset_folder_path, f))])
            print(f'Saved dataset folder"{self.model_flag, "dataset_v" + str(self.num_of_folders + 1)}" with {num_of_files} files.')
            wandb.log({"Dataset folder": f'Saved dataset "{self.model_flag, "dataset_v" + str(self.num_of_folders + 1)}" with {num_of_files} files.'})
            #Create txt file with the number of files, and the file that gave the initial conditions
            with open(os.path.join(os.path.dirname(self.dataset_folder_path), "info.txt"), "w") as text_file:
                text_file.write(f"Number of files: {num_of_files}\n")
                text_file.write(f"Initial conditions file: {self.init_conditions_path}\n")
                text_file.write(f"Number of different simulated trajectories: {self.total_init_conditions}\n")
                #write the time and the number of points
                text_file.write(f"Time horizon of the simulations: {self.time}\n")
                text_file.write(f"Number of points in the each simulation: {self.num_of_points}\n")
        return 


    def load_dataset(self, name):
        """
        Load the dataset.

        Args:
            name (str): The name of the dataset.

        Returns:
            list: The dataset of the synchronous machine.
        """
        dataset_dir = self.config.dirs.dataset_dir
        dataset_path = os.path.join(dataset_dir, self.model_flag, name)
        with open(dataset_path, 'rb') as f:
            dataset = pickle.load(f)
        return dataset

  



    