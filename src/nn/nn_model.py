import torch.nn as nn
import torch
import torch.nn.functional as F
from kan import KAN
import math


class Net(nn.Module):
    """
    A class to represent a neural network model.
    """
    def __init__(self, input_size, hidden_size, output_size):
        super(Net, self).__init__()
        #torch.set_default_dtype(torch.float64)
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = F.tanh()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, hidden_size)
        self.fc5 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        """
        Forward pass of the neural network.
        Args:
            x (torch.Tensor): Input tensor.
        Returns:
            torch.Tensor: Output tensor.
        """
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.relu(self.fc4(x))
        x = self.fc5(x)
        return x

class CosineActivation(nn.Module):
    def forward(self, x):
        return torch.cos(x)

class SwishActivation(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)
    
class Network(nn.Module):
    """
    A class to represent a dynamic neural network model with dynamic number of layers based on the respective argument.
    """
    def __init__(self, input_size, hidden_size, output_size, num_layers,activation="tanh"):
        super(Network, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        if activation=="swish":
            self.activation = SwishActivation()#CosineActivation #nn.Tanh
        elif activation=="cos":
            self.activation = CosineActivation() #nn.Tanh
        elif activation=="tanh":
            self.activation = nn.Tanh()
        self.hidden = []
        self.hidden.append(nn.Linear(self.input_size, self.hidden_size))
        for i in range(self.num_layers):
            self.hidden.append(nn.Linear(self.hidden_size, self.hidden_size))
        self.hidden = nn.ModuleList(self.hidden)
        self.output = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, x):
        """
        Forward pass of the dynamic neural network.
        Args:
            x (torch.Tensor): Input tensor.
        Returns:
            torch.Tensor: Output tensor.
        """
        for i in range(self.num_layers):
            x = self.activation(self.hidden[i](x))
        x = self.output(x)
        return x
    
class Kalm(nn.Module):
    """
    A class to represent a dynamic neural network model with dynamic number of layers based on the respective argument.
    """
    def __init__(self, input_size, hidden_size, output_size, num_layers,grid,k):
        super(Kalm, self).__init__()
        #Fix the size of the NN
        self.size=[input_size]
        for _ in range(num_layers):
            self.size.append(hidden_size)
        self.size.append(output_size)

        self.ka=KAN(self.size,grid=grid, k=k,noise_scale=0.25)#, grid_eps=1.0)
        self.ka.speed()

    def forward(self, x):
        return self.ka(x)
    def update_grid(self,x):
        self.ka.update_grid_from_samples(x)


def _mean_transf(mu, sigma, w, p):
    return torch.exp(-0.5 * (sigma * w) ** 2) * torch.sin(p + mu * w)

def _var_transf(mu, sigma, w, p):
    return (
        0.5
        - 0.5 * torch.exp(-2 * (sigma * w) ** 2) * torch.cos(2 * (p + mu * w))
        - _mean_transf(mu, sigma, w, p) ** 2
    )

def variance_scaling_(tensor, scale=1.0, mode='fan_in', distribution='uniform'):
    fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(tensor)
    if mode == 'fan_in':
        fan = fan_in
    elif mode == 'fan_out':
        fan = fan_out
    elif mode == 'fan_avg':
        fan = (fan_in + fan_out) / 2.0
    else:
        raise ValueError(f"Invalid mode {mode}, expected 'fan_in', 'fan_out', or 'fan_avg'")
    std = math.sqrt(scale / fan)
    if distribution == 'truncated_normal':
        nn.init.normal_(tensor, 0.0, std)
    elif distribution == 'normal':
        nn.init.normal_(tensor, 0.0, std)
    elif distribution == 'uniform':
        bound = math.sqrt(3.0) * std
        nn.init.uniform_(tensor, -bound, bound)
    else:
        raise ValueError(f"Invalid distribution {distribution}, expected 'truncated_normal', 'normal', or 'uniform'")

class ActLayer(nn.Module):
    def __init__(
        self,
        input_dim,
        out_dim,
        num_freqs,
        use_bias=True,
        freeze_basis=False,
        freq_scaling=True,
        freq_scaling_eps=1e-3,
    ):
        super(ActLayer, self).__init__()
        self.input_dim = input_dim
        self.out_dim = out_dim
        self.num_freqs = num_freqs
        self.use_bias = use_bias
        self.freeze_basis = freeze_basis
        self.freq_scaling = freq_scaling
        self.freq_scaling_eps = freq_scaling_eps

        # Initialize trainable parameters
        self.freqs = nn.Parameter(torch.empty(1, 1, num_freqs))
        nn.init.normal_(self.freqs, mean=0.0, std=1.0)
        self.phases = nn.Parameter(torch.empty(1, 1, num_freqs))
        nn.init.zeros_(self.phases)
        self.beta = nn.Parameter(torch.empty(num_freqs, out_dim))
        variance_scaling_(self.beta, scale=1.0, mode='fan_in', distribution='uniform')
        self.lamb = nn.Parameter(torch.empty(input_dim, out_dim))
        variance_scaling_(self.lamb, scale=1.0, mode='fan_in', distribution='uniform')
        if self.use_bias:
            self.bias = nn.Parameter(torch.empty(out_dim))
            nn.init.zeros_(self.bias)

    def forward(self, x):
        # x has shape (batch_size, input_dim)
        if self.freeze_basis:
            freqs = self.freqs.detach()
            phases = self.phases.detach()
        else:
            freqs = self.freqs
            phases = self.phases

        # Compute the basis expansion in one shot:
        # Instead of unsqueezing and then performing a matmul, we compute
        # sin(freqs*x + phases) where x is broadcasted to (B, I, num_freqs)
        # and then fuse the multiplication by beta and lamb with an einsum.
        # Here, x_sin has shape (B, I, F)
        x_sin = torch.sin(x.unsqueeze(-1) * freqs + phases)

        if self.freq_scaling:
            # Compute scaling factors (mean and variance) from freqs and phases.
            # If freqs and phases are not changing (or freeze_basis is True)
            # you could precompute and cache these.
            mu = 0.0
            sigma = 1.0
            mean = _mean_transf(mu, sigma, freqs, phases)
            var = _var_transf(mu, sigma, freqs, phases)
            x_sin = (x_sin - mean) / torch.sqrt(self.freq_scaling_eps + var)

        # Fuse the matmul with the element-wise multiplication.
        # x_sin: (B, I, F), beta: (F, O), lamb: (I, O)
        # The following einsum does:
        #   for each batch element b and output o:
        #       output[b, o] = sum_{i,f} x_sin[b,i,f] * beta[f,o] * lamb[i,o]
        x = torch.einsum("bif,fo,io->bo", x_sin, self.beta, self.lamb)

        if self.use_bias:
            x = x + self.bias

        return x

    # def forward(self, x):
    #     # x should be shape (batch_size, input_dim)
    #     if self.freeze_basis:
    #         freqs = self.freqs.detach()
    #         phases = self.phases.detach()
    #     else:
    #         freqs = self.freqs
    #         phases = self.phases

    #     # Perform basis expansion
    #     x = x.unsqueeze(2)  # shape: (batch_size, input_dim, 1)
    #     x = torch.sin(freqs * x + phases)  # shape: (batch_size, input_dim, num_freqs)

    #     if self.freq_scaling:
    #         mu = 0.0
    #         sigma = 1.0
    #         mean = _mean_transf(mu, sigma, freqs, phases)
    #         var = _var_transf(mu, sigma, freqs, phases)
    #         x = (x - mean) / torch.sqrt(self.freq_scaling_eps + var)

    #     # Optimized computation without constructing 'aux'
    #     # Compute x_beta: (batch_size, input_dim, out_dim)
    #     x_beta = torch.matmul(x, self.beta)  # beta: (num_freqs, out_dim)
    #     # Multiply element-wise with lambdas
    #     x_beta_lamb = x_beta * self.lamb.unsqueeze(0)  # lamb: (input_dim, out_dim)
    #     # Sum over input_dim
    #     x = x_beta_lamb.sum(dim=1)  # shape: (batch_size, out_dim)

    #     if self.use_bias:
    #         x = x + self.bias  # shape: (batch_size, out_dim)

    #     return x  # shape: (batch_size, out_dim)

class ActNet(nn.Module):
    def __init__(
        self,
        input_dim,
        embed_dim,
        num_layers,
        out_dim,
        num_freqs,
        output_activation=None,
        w0_init=30.0,
        w0_fixed=False,
        use_act_bias=True,
        freeze_basis=False,
        freq_scaling=True,
        freq_scaling_eps=1e-3,
    ):
        super(ActNet, self).__init__()
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.out_dim = out_dim
        self.num_freqs = num_freqs
        self.output_activation = output_activation
        self.w0_fixed = w0_fixed
        self.use_act_bias = use_act_bias
        self.freeze_basis = freeze_basis
        self.freq_scaling = freq_scaling
        self.freq_scaling_eps = freq_scaling_eps

        # Initialize w0 parameter
        if not self.w0_fixed:
            self.w0 = nn.Parameter(torch.tensor(w0_init))
        else:
            self.register_buffer('w0', torch.tensor(w0_fixed))

        # Projection layer
        self.proj = nn.Linear(input_dim, embed_dim, bias=True)
        # Initialize projection bias
        nn.init.uniform_(self.proj.bias, -math.sqrt(3), math.sqrt(3))

        # Build layers
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            act_layer = ActLayer(
                input_dim=embed_dim,
                out_dim=embed_dim,
                num_freqs=num_freqs,
                use_bias=use_act_bias,
                freeze_basis=freeze_basis,
                freq_scaling=freq_scaling,
                freq_scaling_eps=freq_scaling_eps,
            )
            self.layers.append(act_layer)

        # Output layer
        self.output_layer = nn.Linear(embed_dim, out_dim)
        # Initialize output layer weights
        nn.init.kaiming_uniform_(self.output_layer.weight, a=math.sqrt(5))
        if self.output_layer.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.output_layer.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.output_layer.bias, -bound, bound)

    def forward(self, x):
        # x has shape (batch_size, input_dim)
        if not self.w0_fixed:
            w0 = F.softplus(self.w0)
        else:
            w0 = self.w0
        x = x * w0
        x = self.proj(x)

        for i in range(self.num_layers):
            x = self.layers[i](x)

        x = self.output_layer(x)
        if self.output_activation is not None:
            x = self.output_activation(x)
        return x


class PinnA(nn.Module): # DISCARD IT, OUTPUT IS WRONG
    """
    A class to represent a Pinn model with adjusted output.
    """
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(PinnA, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.hidden = []
        self.hidden.append(nn.Linear(self.input_size, self.hidden_size))
        for i in range(self.num_layers):
            self.hidden.append(nn.Linear(self.hidden_size, self.hidden_size))
        self.hidden = nn.ModuleList(self.hidden)
        self.output = nn.Linear(self.hidden_size, self.output_size)
        #self.shortcut = nn.Linear(self.input_size, self.output_size)

    def forward(self, x):
        """
        Forward pass of the PinnA model.
        Args:
            x (torch.Tensor): Input tensor.
        Returns:
            torch.Tensor: Output tensor.
        """
        y = torch.tanh(self.hidden[0](x))
        for i in range(self.num_layers-1):
            y = torch.tanh(self.hidden[i+1](y))
        y = self.output(y)
        time = x[:,0].view(-1,1)
        #time = torch.where(time < 0.5, time, 0.5*torch.ones_like(time))
        y = x[:,1:] + y*time
        return y
    

class ResidualBlock(nn.Module):
    """
    A class to represent a residual block in a fully connected ResNet model.
    """
    def __init__(self, in_features, out_features, activation=nn.ReLU()):
        super(ResidualBlock, self).__init__()
        self.fc1 = nn.Linear(in_features, out_features)
        self.activation = activation
        self.fc2 = nn.Linear(out_features, out_features)

    def forward(self, x):
        identity = x
        out = self.fc1(x)
        out = self.activation(out)
        out = self.fc2(out)
        out += identity
        out = self.activation(out)
        return out

class FullyConnectedResNet(nn.Module):
    """
    A class to represent a fully connected ResNet model.
    """
    def __init__(self, input_size, hidden_size, output_size, num_blocks, num_layers_per_block, activation=nn.ReLU()):
        super(FullyConnectedResNet, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_blocks = num_blocks
        self.num_layers_per_block = num_layers_per_block
        self.activation = activation

        self.fc_input = nn.Linear(input_size, hidden_size)
        self.blocks = self._make_blocks()
        self.fc_output = nn.Linear(hidden_size, output_size)

    def _make_blocks(self):
        blocks = []
        for _ in range(self.num_blocks):
            block_layers = []
            in_features = self.hidden_size
            for _ in range(self.num_layers_per_block):
                block_layers.append(ResidualBlock(in_features, self.hidden_size, self.activation))
                in_features = self.hidden_size
            blocks.append(nn.Sequential(*block_layers))
        return nn.ModuleList(blocks)

    def forward(self, x):
        x = self.fc_input(x)
        for block in self.blocks:
            x = block(x)
        x = self.fc_output(x)
        return x




