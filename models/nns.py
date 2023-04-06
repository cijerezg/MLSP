import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Categorical
from typing import List, Type
import pdb
import torch.autograd as autograd


def create_mlp(
    input_dim: int,
    output_dim: int,
    net_arch: List[int],
    activation_fn: Type[nn.Module] = nn.ReLU,
    use_batch_norm: bool = False,
    squash_output: bool = False,
) -> List[nn.Module]:
    """Create a multilayer perceptron (MLP).

    An MLP is a collection of fully-connected layers (linear
    transformations) followed by an activation function.

    Parameters
    ----------
    input_dim : int
        Dimension of the input vector
    output_dim : int
        Dimension of the output vector
    net_arch : List[int]
        Architecture of the neural net. It represents the number of
        units per layer. The length of this list is the number of
        layers.
    activation_fn : Type[nn.Module]
        The activation function to use after each layer.
    use_batch_norm : bool
        Whether to include a batch norm layer.
    squash_output : bool
        Whether to squash the output using a Tanh activation function.

    Returns
    -------
    List[nn.Module]
        List containing all layers, including activation functions.
    """
    if len(net_arch) > 0:
        modules = [nn.Linear(input_dim, net_arch[0]), activation_fn()]
    else:
        modules = []

    for idx in range(len(net_arch) - 1):
        modules.append(nn.Linear(net_arch[idx], net_arch[idx + 1]))
        if use_batch_norm:
            modules.append(nn.BatchNorm1d(num_features=net_arch[idx]))
        modules.append(activation_fn())

    if output_dim > 0:
        last_layer_dim = net_arch[-1] if len(net_arch) > 0 else input_dim
        modules.append(nn.Linear(last_layer_dim, output_dim))
    if squash_output:
        modules.append(nn.Tanh())
    return modules


LOG_STD_MAX = 2
LOG_STD_MIN = -20


class Encoder(nn.Module):
    def __init__(self, H_dim, input_dim, latent_dim=12, lstm_hidden=128, enc_size=64):
        super().__init__()
        self.embed = nn.Linear(input_dim, lstm_hidden)
        self.lstm = nn.LSTM(input_size=lstm_hidden, hidden_size=lstm_hidden, batch_first=True)

        self.out = nn.Linear(lstm_hidden, enc_size)
        self.fc_mu = nn.Linear(enc_size, latent_dim)
        self.fc_log_std = nn.Linear(enc_size, latent_dim)
        self.lstm_hidden = lstm_hidden
        self.latent_dim = latent_dim

    def forward(self, x):
        # x: [batch, seq_len, input_size]
        batch_size, input_size = x.shape[0], x.shape[-1]
        x = x.reshape(-1, input_size)  # [batch*seq_len, input_size]
        embed = self.embed(x)
        embed = embed.reshape(batch_size, -1, self.lstm_hidden)  # [batch, seq_len, lstm_hidden]

        o, _ = self.lstm(embed)  # [batch, seq_len, lstm_hidden]
        o = o[:, -1, :]  # take last output cell

        o = F.relu(self.out(o))
        mu = self.fc_mu(o)
        log_std = self.fc_log_std(o)
        std = torch.exp(torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX))
        
        density = Normal(mu, std)
        sample = density.rsample()

        return sample, density, mu, std


class Decoder(nn.Module):
    def __init__(self, H_dim, output_dim, input_dim=12, lstm_hidden=128, dec_size=64):
        super().__init__()
        self.lstm_hidden = lstm_hidden
        self.output_dim = output_dim
        self.H_dim = H_dim

        self.embed = nn.Linear(input_dim, lstm_hidden)  # input_dim=latent_dim
        self.lstm = nn.LSTM(input_size=lstm_hidden, hidden_size=lstm_hidden, batch_first=True)

        self.out = nn.Linear(lstm_hidden, dec_size)
        self.action_layer = nn.Linear(dec_size, output_dim)

    def forward(self, x):
        # x: [batch, latent_dim]
        x = x.view(x.shape[0], 1, x.shape[-1]).repeat(1, self.H_dim, 1)  # x: [batch, seq_len, latent_dim]
        batch_size, _, input_size = x.shape
        x = x.reshape(-1, input_size)  # [batch*seq_len, input_size]
        embed = self.embed(x)  # [batch*seq_len, lstm_hidden]
        embed = embed.reshape(batch_size, -1, self.lstm_hidden)
        o, _ = self.lstm(embed)
        o = o.reshape(-1, self.lstm_hidden)
        o = F.relu(self.out(o))
        action = self.action_layer(o)
        action = action.reshape(batch_size, self.H_dim, -1)

        return action

class SkillPrior(nn.Module):
    def __init__(self, input_obs, input_l=None, net_arch=[128] * 6, latent_dim=12):
        super().__init__()
        
        self.embed_obs = nn.Linear(input_obs, 128)
        if input_l is not None:
            self.embed_l = nn.Linear(input_l, 128)
        latent_policy = create_mlp(128, output_dim=-1,
                                   net_arch=net_arch, use_batch_norm=False)
        self.latent_policy = nn.Sequential(*latent_policy)
        self.mu = nn.Linear(net_arch[-1], latent_dim)
        self.log_std = nn.Linear(net_arch[-1], latent_dim)
        self.input_obs = input_obs
        self.input_l = input_l
    
    def forward(self, obs):
        x = self.embed_obs(obs[:, :self.input_obs])
        if self.input_l is not None:
            try:
                leng = self.embed_l(obs[:, self.input_obs:self.input_obs + self.input_l])
            except RuntimeError:
                pdb.set_trace()
            x = x + leng
        policy_latent = self.latent_policy(x)
        mu = self.mu(policy_latent)
        log_std = self.log_std(policy_latent)
        std = torch.exp(torch.clamp(log_std, min=LOG_STD_MIN, max=LOG_STD_MAX))
        
        density = Normal(mu, std)
        
        return density

class LengthPrior(nn.Module):
    def __init__(self, input_dim, net_arch=[128] * 6, lengths=3):
        super().__init__()
        latent_policy = create_mlp(input_dim, output_dim=-1, net_arch=net_arch, use_batch_norm=False)
        self.latent_policy = nn.Sequential(*latent_policy)
        self.midway = nn.Linear(net_arch[-1], 32)
        self.out = nn.Linear(32, lengths)

    def forward(self, observation):
        policy_latent = self.latent_policy(observation)
        policy_latent = F.relu(self.midway(policy_latent))
        prior = F.log_softmax(self.out(policy_latent), dim=1)
        return prior



class LengthPolicy(nn.Module):
    def __init__(self, input_dim, lengths, net_arch=[128] * 6):
        super().__init__()
        latent_policy = create_mlp(input_dim, -1, net_arch=net_arch, use_batch_norm=False)
        self.latent_policy = nn.Sequential(*latent_policy)

        self.preout = nn.Linear(net_arch[-1], 32)
        self.out = nn.Linear(32, lengths)

    def forward(self, obs):
        policy_latent = self.latent_policy(obs)
        z_l = F.relu(self.preout(policy_latent))
        length_prior = F.log_softmax(self.out(z_l), dim=1)

        dist = Categorical(logits=length_prior)
        samples = dist.sample()

        return length_prior, samples
        


class SkillPolicy(nn.Module):
    def __init__(self, input_obs, input_l, action_range=4, net_arch=[128] * 3, latent_dim=12):
        super().__init__()
        self.embed_obs = nn.Linear(input_obs, 128)
        if input_l is not None:
            self.embed_l = nn.Linear(input_l, 128)
                    
        latent_policy = create_mlp(128, -1, net_arch=net_arch, use_batch_norm=False)
        self.latent_policy = nn.Sequential(*latent_policy)
        latent_policy2 = create_mlp(128, -1, net_arch=net_arch, use_batch_norm=False)
        self.latent_policy2 = nn.Sequential(*latent_policy2)

        self.mu = nn.Linear(net_arch[-1], latent_dim)
        self.log_std = nn.Linear(net_arch[-1], latent_dim)

        self.input_obs = input_obs
        self.input_l = input_l
        self.action_range = action_range

    def forward(self, data):
        x = self.embed_obs(data[:, 0:self.input_obs])
        policy_latent = self.latent_policy(x)
        
        if self.input_l is not None:
            embed_l = self.embed_l(data[:, self.input_obs:])
            policy_latent = policy_latent + embed_l
        
        policy_latent = self.latent_policy2(policy_latent)
        
        mu = self.mu(policy_latent)

        log_std = self.log_std(policy_latent)
        std = torch.exp(torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX))
        
        density = Normal(mu, std)
        sample = density.rsample()
        sample = self.action_range * torch.tanh(sample / self.action_range)
        
        return sample, density, mu, std
    

class Critic(nn.Module):
    def __init__(self, obs_dim, z_dim, l_dim=None, net_arch=[256] * 2):
        super().__init__()
        self.embed_obs = nn.Linear(obs_dim, 128)
        self.embed_z = nn.Linear(z_dim, 128)
        if l_dim is not None:
            self.embed_l = nn.Linear(l_dim, 256)
        
        self.latent_policy = create_mlp(128, -1, net_arch=net_arch)
        self.latent_policy = nn.Sequential(*self.latent_policy)
        self.latent_policy2 = create_mlp(256, -1, net_arch=net_arch)
        self.latent_policy2 = nn.Sequential(*self.latent_policy2)
        self.out = nn.Linear(32, 1)
        self.obs_dim = obs_dim
        self.z_dim = z_dim
        self.l_dim = l_dim

        self.post_pol = nn.Linear(net_arch[-1], 32)
        

    def forward(self, data):
        e_obs = self.embed_obs(data[:, :self.obs_dim])
        e_z = self.embed_z(data[:, self.obs_dim:self.obs_dim + self.z_dim])
        x = e_obs + e_z

        qvalue = self.latent_policy(x)
        if self.l_dim is not None:
            e_l = self.embed_l(data[:, self.obs_dim + self.z_dim: self.obs_dim + self.z_dim + self.l_dim])
            qvalue = qvalue + e_l

        qvalue = self.latent_policy2(qvalue)
        qvalue = F.relu(self.post_pol(qvalue))
        qvalue = self.out(qvalue)

        return qvalue
