import torch
import torch.nn as nn

class PINNWeighting:
    def __init__(self, model, cfg, device, loss_dimension, wandb_run, beta=0.99):
        self.model = model
        self.weights = cfg.nn.weighting.weights # Weights will be initialized based on number of residuals dynamically
        self.scheme = cfg.nn.weighting.update_weight_method
        self.update_weights_freq = cfg.nn.weighting.update_weights_freq
        self.beta = beta
        self.wandb_run = wandb_run
        self.loss_dimension = (1 if (cfg.nn.weighting.flag_mean_weights or cfg.nn.time_factored_loss) else loss_dimension)
        self.device = device
        self.initialize_weights()
        self.epoch_flag = -1

    def initialize_weights(self):
        # check that the weights are valid and positive
        if self.weights is not None:
            if len(self.weights) != 4:
                raise ValueError("Weights must be a list of 4 values: [loss_data, loss_dt, loss_pinn, loss_pinn_ic]")
            if any(weight < 0 for weight in self.weights):
                raise ValueError("Weights must be positive")
        else:
            raise ValueError("Correct Weights must be provided")
        print("Weights initialized as: ", self.weights," are updated with scheme: ", self.scheme, ("every " + str(self.update_weights_freq) + " epochs") if self.scheme!="Static" else "")
        # Initialize weights if not already initialized
        data_term = 1
        col_term = 0.1
        if self.weights is None:
            total_losses = 1 + self.loss_dimension+ self.loss_dimension + 1 # 1 for loss_data, 1 for loss_pinn_ic and self.loss_dimension for each loss_dt and loss_pinn
            self.weights = torch.ones(total_losses, device=self.device, dtype=torch.float32, requires_grad=(self.scheme == 'Sam'))
            # i want to have a balancing factor for the loss terms with 1, 1,0.1,0.1 for loss_data, loss_dt, loss_pinn, loss_pinn_ic
            self.balancing_term = torch.tensor([1,0.01,0.01,0.01], device=self.device, dtype=torch.float32, requires_grad=False)
        else:
            # Update weights using the scheme and the count of loss components
            updated_weights = (
                [self.weights[0]] +  # For loss_data
                [self.weights[1]] * (self.loss_dimension) +  # For each term in loss_dt
                [self.weights[2]] * (self.loss_dimension) +  # For each term in loss_pinn
                [self.weights[3]]  # For loss_pinn_ic
            )
            self.weights = torch.tensor(updated_weights, device=self.device, dtype=torch.float32, requires_grad=(self.scheme == 'Sam'))
            # i want to have a balancing factor for the loss terms with 1, 1,0.1,0.1 for loss_data, loss_dt, loss_pinn, loss_pinn_ic, maybe i have multiple weights thoug...
            
            balancing_term = ([1] + [0.01] * (self.loss_dimension) + [0.01] * (self.loss_dimension) + [0.01])
            self.balancing_term = torch.tensor(balancing_term, device=self.device, dtype=torch.float32, requires_grad=False)

        # Initialize weight mask
        self.weight_mask = torch.where(self.weights == 0,torch.tensor(0.0, device=self.weights.device),torch.tensor(1.0, device=self.weights.device))

        # If using Sam scheme, ensure weights are parameters
        if self.scheme == 'Sam':
            self.soft_adaptive_weights = nn.Parameter(self.weights)
            self.weights = self.soft_adaptive_weights
        self.log_weights(epoch = 0)
        return


    def compute_weighted_loss(self, loss_data, loss_dt, loss_pinn, loss_pinn_ic,epoch):#, iteration_count=None
        # Calculate losses
        # If loss_dimension is 1, then we need to take mean of the losses
        if self.loss_dimension == 1:
            loss_dt = loss_dt.mean()#torch.mean(torch.stack(loss_dt))
            loss_pinn = loss_pinn.mean()#torch.mean(torch.stack(loss_pinn))

        #SOS multiply with loss_dimension to balance it
        loss_data_ = self.weights[0]  * loss_data * self.loss_dimension
        loss_dt_ = [self.weights[1+i] * (loss_dt[i] if self.loss_dimension > 1 else loss_dt) for i in range(self.loss_dimension)]
        loss_pinn_ = [self.weights[1+self.loss_dimension+i] * (loss_pinn[i] if self.loss_dimension > 1 else loss_pinn) for i in range(self.loss_dimension)]
        loss_pinn_ic_ = self.weights[-1] * loss_pinn_ic * self.loss_dimension
        individual_weighted_losses = [loss_data_] + loss_dt_ + loss_pinn_ + [loss_pinn_ic_]
        #individual_weighted_losses = [torch.tensor(0.0, device=self.device) if torch.isnan(loss) else loss for loss in individual_weighted_losses]
        
        # Calculate total loss
        self.total_loss = torch.stack(individual_weighted_losses).sum()        
        # Update weights with the desired frequency, iteration_count is used to check if the update is needed and not epoch due to lbfgs internal iterations
        if (epoch + 1) % self.update_weights_freq == 0 and self.scheme != 'Sam': 
            if epoch != self.epoch_flag: # To avoid multiple updates in the same epoch
                self.update_weights(individual_weighted_losses, epoch)
                #print("Updated Weights", self.weights.tolist())
                self.epoch_flag = epoch
        return self.total_loss, individual_weighted_losses


    def update_weights(self, losses, epoch):
        if self.scheme == 'Gradient':
            # Gradient-based weighting update
            grad_norms = []
            for loss in losses:
                #check if weight_mask is 0 or not
                if self.weight_mask[len(grad_norms)] == 0:
                    grad_norms.append(0)
                    continue
                self.model.zero_grad()
                loss.backward(retain_graph=True)
                grad_norm = torch.norm(torch.stack([torch.norm(p.grad) for p in self.model.parameters() if p.grad is not None]))
                # Check if grad_norm is NaN and skip if it is
                if torch.isnan(grad_norm):
                    print("Warning: NaN gradient norm detected.")
                    grad_norms.append(torch.tensor(0.0)) # Append 0 to avoid division by zero
                else:
                    grad_norms.append(grad_norm.item())
                
            grad_norms = torch.tensor(grad_norms)
            grad_norms = torch.nan_to_num(grad_norms, nan=0.0)  # Replace NaN with 0

            grad_norms_avg = torch.mean(grad_norms)
            new_weights = grad_norms_avg / (grad_norms + 1e-8)
            #print('grad norms are', grad_norms)
            #print('grad norms avg is', grad_norms_avg)
            #print('New weights are', new_weights)

        elif self.scheme == 'Ntk':
            # NTK-based weighting update
            ntk_traces = []
            for loss in losses:
                if self.weight_mask[len(ntk_traces)] == 0:
                    ntk_traces.append(0)
                    continue
                
                self.model.zero_grad()
                loss.backward(retain_graph=True)
                ntk_trace = 0.0
                for param in self.model.parameters():
                    if param.grad is not None:
                        ntk_trace += torch.sum(param.grad ** 2)
                if torch.isnan(ntk_trace):
                    print("Warning: NaN NTK trace detected.")
                    ntk_traces.append(torch.tensor(0.0)) 
                else:
                    ntk_traces.append(ntk_trace.item())
            
            ntk_traces = torch.tensor(ntk_traces, device=self.device)
            ntk_traces = torch.nan_to_num(ntk_traces, nan=0.0)  # Replace NaN with 0
            ntk_traces_avg = torch.mean(ntk_traces)
            # To avoid division by zero, add a small epsilon
            new_weights = ntk_traces_avg / (ntk_traces + 1e-8)
            #print('ntk norms are', ntk_traces)
            #print('ntk norms avg is', ntk_traces_avg)
            #print('New weights are', new_weights)

        elif self.scheme == 'Sam':
            # SAM-based weighting update
            epsilon = 1e-8
            
            with torch.no_grad():
                self.soft_adaptive_weights += self.weight_mask * 0.1 / (self.soft_adaptive_weights.grad + epsilon)
                self.log_weights(epoch)
                print("0.1* sam",  0.1 / (self.soft_adaptive_weights.grad + epsilon))
                self.soft_adaptive_weights.grad.zero_()
                
                self.weights = self.soft_adaptive_weights 
                print(self.weights)
                
            return
            
        elif self.scheme == 'Static':
            return
        
        
        else:
            raise ValueError("Unknown weighting scheme. Choose either 'gradient' or 'ntk' or 'sam'.")
        
        # Update weights using moving average
        self.weights = ( self.weights * self.weight_mask + self.balancing_term *  new_weights.to(self.weights.device)) * self.weight_mask
        print("new weights 0.01*",new_weights)
        print(self.weights)
        
        self.log_weights(epoch)

        return
    
    def log_weights(self, epoch):
        if self.scheme == 'Sam':
            if self.wandb_run is not None and epoch!=0:
                log_data = {f"weights {i}": self.soft_adaptive_weights[i] for i in range(len(self.soft_adaptive_weights))}
                log_data.update({f"soft_adaptive_weights grad{i}": self.soft_adaptive_weights.grad[i] for i in range(len(self.soft_adaptive_weights))})
                log_data["epoch"] = epoch
                self.wandb_run.log(log_data)

        else:
            if self.wandb_run is not None:
                for i in range(len(self.weights)):
                    self.wandb_run.log({"weights"+str(i): self.weights[i], 'epoch': epoch})
        return
    
    def log_losses(self, loss, epoch, name):

        if self.wandb_run is not None:
            for i in range(len(loss)):
                self.wandb_run.log({name[i]: loss[i], 'epoch': epoch})
        return
