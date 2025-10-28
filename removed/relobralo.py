import torch
import torch.nn as nn
import torch.nn.functional as F

class ReLoBRaLoLoss(nn.Module):
    """
    Class for the ReLoBRaLo Loss. 
    This class extends the nn.Module to have dynamic weighting for each term.
    """
    def __init__(self, alpha=0.999, temperature=0.1, rho=0.99):
        """
        Parameters
        ----------
        alpha, optional : float
            Controls the exponential weight decay rate. 
            Value between 0 and 1. The smaller, the more stochasticity.
            0 means no historical information is transmitted to the next iteration.
            1 means only first calculation is retained. Defaults to 0.999.
        temperature, optional : float
            Softmax temperature coefficient. Controlls the "sharpness" of the softmax operation. 
            Defaults to 0.1.
        rho, optional : float
            Probability of the Bernoulli random variable controlling the frequency of random lookbacks.
            Value berween 0 and 1. The smaller, the fewer lookbacks happen.
            0 means lambdas are always calculated w.r.t. the initial loss values.
            1 means lambdas are always calculated w.r.t. the loss values in the previous training iteration.
            Defaults to 0.99.
        """
        super(ReLoBRaLoLoss, self).__init__()
        self.alpha = alpha
        self.temperature = temperature
        self.rho = rho
        self.call_count = torch.tensor(0, dtype=torch.int16)
        self.EPS = 1e-8  # Small constant to avoid division by zero

        self.lambdas = None
        self.last_losses = None
        self.init_losses = None

    def forward(self, data_loss, data_dt_loss, pinn_loss, pinn_loss_ic, weight_data, weight_dt, weight_pinn, weight_pinn_ic):
        losses = [data_loss, data_dt_loss, pinn_loss, pinn_loss_ic]
        if self.init_losses is None:
            self.init_losses = torch.tensor([1. for _ in range(len(losses))], device=losses[0].device)
        if self.last_losses is None:
            self.last_losses = torch.tensor([1. for _ in range(len(losses))], device=losses[0].device)
        self.lambdas = torch.tensor([weight_data, weight_dt, weight_pinn, weight_pinn_ic], device=losses[0].device)
        
        # Calculate alpha and rho
        alpha = torch.where(self.call_count == 0, torch.tensor(1., device=losses[0].device), 
                             torch.where(self.call_count == 1, torch.tensor(0., device=losses[0].device), 
                                         torch.tensor(self.alpha, device=losses[0].device)))
        rho = torch.where(self.call_count == 0, torch.tensor(1., device=losses[0].device), 
                           torch.where(self.call_count == 1, torch.tensor(1., device=losses[0].device), 
                                       (torch.rand(1, device=losses[0].device) < self.rho).float()))

        # Compute new lambdas w.r.t. the losses in the previous iteration
        lambdas_hat = [(losses[i] / (self.last_losses[i] * self.temperature + self.EPS)) for i in range(len(losses))]
        lambdas_hat =  torch.max(torch.tensor(lambdas_hat).cpu())- F.softmax(torch.tensor(lambdas_hat).cpu(), dim=0) 
        lambdas_hat = lambdas_hat.detach().cpu().numpy() * len(losses)

        # Compute new lambdas w.r.t. the losses in the first iteration
        
        init_lambdas_hat = [(losses[i] / (self.init_losses[i] * self.temperature + self.EPS)) for i in range(len(losses))]
        init_lambdas_hat = torch.max(torch.tensor(init_lambdas_hat).cpu())-F.softmax(torch.tensor(init_lambdas_hat).cpu(), dim=0)
        init_lambdas_hat = init_lambdas_hat.detach().cpu().numpy() * len(losses)

        # Use rho for deciding, whether a random lookback should be performed
        new_lambdas = [(rho * alpha * self.lambdas[i] + (1 - rho) * alpha * init_lambdas_hat[i] + (1 - alpha) * lambdas_hat[i]) for i in range(len(losses))]
        self.lambdas = new_lambdas

        self.call_count += 1
        self.last_losses = losses

        return self.lambdas



class LossWeightScheduler:
    def __init__(
        self,
        nn_model,
        max_value: float = 0.0,
        epochs_to_tenfold: int = 20,
        initial_value: float = 1.0e-7,
    ):
        self.max_value = torch.tensor(max_value)
        assert epochs_to_tenfold > 0
        self.epoch_factor = torch.tensor(10.0) ** (1 / epochs_to_tenfold)
        self.current_value = (
            torch.tensor(initial_value) if max_value > 0.0 else torch.tensor(0.0)
        )
        self.nn_model = nn_model

    def __call__(self):
        self.current_value = torch.minimum(
            self.current_value * self.epoch_factor, self.max_value
        )
        self.nn_model.physics_regulariser = self.current_value