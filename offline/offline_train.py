"""Define all offline training, e.g., decoder, encoder priors."""

import sys
sys.path.insert(0, '../')

import os
import numpy as np
import torch
from torch.func import functional_call
from torch.distributions.kl import kl_divergence
from torch.distributions import Normal
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from utilities.optimization import Adam_update
from utilities.utils import hyper_params
from models.nns import SkillPrior, LengthPrior
from models.nns import Encoder, Decoder
import wandb
import gym
import d4rl
import pdb

class HIVES(hyper_params):
    """Define all offline and train all offline models."""
    def __init__(self, config):
        """Config contain all VAE hyper parameters."""
        super().__init__(config)
    
        self.models = {}
        self.names = []

        for idx, level in enumerate(self.hrchy_full):
            if idx == 0:
                io_dim = self.action_dim
            else:
                m = self.level_lengths_full[idx - 1]
                io_dim = self.hrchy_full[m]['z']

            encoder = Encoder(self.hrchy_full[level]['length'],
                              input_dim=io_dim,
                              latent_dim=self.hrchy_full[level]['z']).to(self.device)

            decoder = Decoder(self.hrchy_full[level]['length'],
                              output_dim=io_dim,
                              input_dim=self.hrchy_full[level]['z']).to(self.device)

            self.models[f'Encoder{level}'] = encoder
            self.models[f'Decoder{level}'] = decoder

            self.names.extend([f'Encoder{level}', f'Decoder{level}'])

        if self.length_dim is not None:
            self.models['LengthPrior'] = LengthPrior(
                self.state_dim, lengths=self.length_dim).to(self.device)
            self.names.extend(['LengthPrior'])

        self.models['SkillPrior'] = SkillPrior(
            self.state_dim, input_l=self.length_dim,
            latent_dim=self.lat_dim).to(self.device)

        self.names.extend(['SkillPrior'])

    def dataset_loader(self):
        """Dataset is one single file."""
        self.load_dataset()
        
        dset_train = Drivedata(self.indexes)

        self.loader = DataLoader(dset_train, shuffle=True, num_workers=8,
                                 batch_size=self.offline_batch_size)

    def train_vae_level(self, params, optimizers, beta, level):
        """Train a particular encoder and decoder level.

        It may use previous levels, but those parameters are not
        trained.
        """
        for i, idx in enumerate(self.loader):
            action = torch.from_numpy(self.dataset['actions'][idx])
            action = action.view(-1, action.shape[-1]).to(self.device)
            recon_loss, kl_loss = self.vae_loss(action, params, level, i)
            loss = recon_loss + beta * kl_loss
            m = self.level_lengths[level]
            losses = [loss]
            params_names = [f'Level{m}']
            params = Adam_update(params, losses, params_names, optimizers)

        wandb.log({'VAE/loss': recon_loss.detach().cpu()})
        wandb.log({'VAE/kl_loss': kl_loss.detach().cpu()})
        
        return params

    def vae_loss(self, action, params, level, i):
        """VAE loss."""
        z_seq, pdf, mu, std = self.evaluate_encoder_hrchy(action, params, level)
        rec = self.evaluate_decoder_hrchy(z_seq, params, level)

        error = torch.square(action - rec).mean(1)
        rec_loss = -Normal(rec, 1).log_prob(action).sum(axis=-1)
        rec_loss = rec_loss.mean()

        if i == 0:
            wandb.log({'VAE/[encoder] STD':
                       wandb.Histogram(std.detach().cpu())})
            wandb.log({'VAE/[encoder] Mu':
                       wandb.Histogram(mu.detach().cpu())})
            wandb.log({'VAE/[decoder]reconstruction_std': rec.std(0).mean().detach().cpu()})
            if rec_loss < 12:
                wandb.log({'VAE/MSE Distribution':
                           wandb.Histogram(error.detach().cpu())})

        N = Normal(0, 1)
        kl_loss = torch.mean(kl_divergence(pdf, N))

        return rec_loss, kl_loss

    def evaluate_encoder_hrchy(self, mu, params, level):
        """Evaluate encoder up to a certain level."""
        m_sup = self.level_lengths[level]
        iters = self.mapper[m_sup]
        
        for i in range(iters + 1):
            m = self.level_lengths_full[i]
            mu = mu.reshape(-1, self.hrchy_full[m]['length'], mu.shape[-1])
            z, pdf, mu, std = functional_call(self.models[f'Encoder{m}'],
                                              params[f'Encoder{m}'], mu)

        return z, pdf, mu, std

    def evaluate_decoder_hrchy(self, rec, params, level):
        """Evaluate decoder using a the given number of levels.

        The output will always be an action.
        """
        if torch.is_tensor(level):
            level = level.cpu().numpy()        
            m_sup = self.level_lengths[level[0]]

        else:
            m_sup = self.level_lengths[level]

        iters = self.mapper[m_sup]
        
        for i in range(iters, -1, -1):
            m = self.level_lengths_full[i]
            rec = functional_call(self.models[f'Decoder{m}'],
                                  params[f'Decoder{m}'], rec)
            rec = rec.reshape(-1, rec.shape[-1])
            
        return rec

    def set_skill_lookup(self, levels, params):
        """Save all skils to train priors."""
        all_actions = torch.from_numpy(self.dataset['actions']).to(self.device)
        a, b, _ = all_actions.shape
        self.weights = torch.zeros(a, len(self.hrchy)).to(self.device)
        self.loc = torch.zeros(a, len(self.hrchy), self.lat_dim).to(self.device)
        self.scale = torch.zeros(a, len(self.hrchy), self.lat_dim).to(self.device)

        bs_size = 1024
        number_of_batches = all_actions.shape[0] // bs_size + 1

        for j in range(number_of_batches):
            actions = all_actions[j * bs_size:(j + 1) * bs_size, :, :]
            for i in range(levels):
                with torch.no_grad():
                    _, pdf, z, _ = self.evaluate_encoder_hrchy(actions, params, i)
                    rec = self.evaluate_decoder_hrchy(z, params, i)
                rec = rec.reshape(-1, self.max_length, rec.shape[-1])
                mse = torch.mean((actions - rec)**2, dim=2)
                weight = mse.mean(1)
                self.weights[j * bs_size: (j + 1) * bs_size, i] = weight
                loc, scale = pdf.loc, pdf.scale
                loc, scale = self.resize_level_pdf(loc, i), self.resize_level_pdf(scale, i)
                self.loc[j * bs_size: (j + 1) * bs_size, i, :] = loc
                self.scale[j * bs_size: (j + 1) * bs_size, i, :] = scale

        vecs = [val for val in self.level_lengths.values()]  # Get all the lengths, e.g., 8, 32, 64
        vecs = torch.tensor(vecs).to(self.device)
        self.vecs = vecs.reshape(1, -1)
        
    def reshape_level(self, x, level):
        """Reshape tensors to compute length prior."""
        x = x.reshape(-1, self.max_length)
        return x[:, 0:self.level_lengths[level]]

    def resize_level_pdf(self, x, level):
        """Reshape tensors to compute skill prior."""
        lev_len = self.level_lengths[level]
        x = x.reshape(-1, self.skill_length // lev_len, x.shape[-1])
        return x[:, 0, :]
        
    def train_prior(self, params, optimizers, length=True):
        """Trains one epoch of length prior."""
        for i, idx in enumerate(self.loader):
            obs = self.dataset['observations'][idx][:, 0, :]
            obs = torch.from_numpy(obs).to(self.device)
            if length:
                prior_loss = self.length_prior_loss(idx, obs, params, i)
            else:
                prior_loss = self.skill_prior_loss(idx, obs, params, i)
            name = ['LengthPrior'] if length else ['SkillPrior']
            loss = [prior_loss]
            params = Adam_update(params, loss, name, optimizers)

        return params
    
    def length_prior_loss(self, idx, obs, params, i):
        """Compute loss for length prior."""
        probs = functional_call(self.models['LengthPrior'],
                                params['LengthPrior'],
                                obs)
    
        weights = self.weights[idx, :]

        normed_weights = weights / weights.mean(0).reshape(1, -1)
        nu = 4 if self.env_key == 'kitchen' else 12

        softmax_args = nu * normed_weights
        target_dist = F.log_softmax(-softmax_args, dim=1)

        imp_loss = F.kl_div(probs, target_dist, log_target=True, reduction='batchmean')

        if i == 0:
            wandb.log({'LengthPrior/max_prob_mean': torch.max(probs, dim=1)[0].mean().detach().cpu()})
            max_probs = torch.max(probs, dim=1)[1].to(torch.float16)
            max_probs_target = torch.max(target_dist, dim=1)[1].to(torch.float16)
            for i in range(probs.shape[1]):
                wandb.log({f'LengthPrior/Percentage {i}': sum(max_probs == i) / max_probs.shape[0]})
                wandb.log({f'LengthPrior/Percentage target dist {i}':
                           sum(max_probs_target == i) / max_probs_target.shape[0]})
            wandb.log({'LengthPrior/trade-off_loss': imp_loss.detach().cpu()})
            wandb.log({'LengthPrior/target_dist max': torch.max(target_dist, dim=1)[0].mean().detach().cpu()})
            
        return imp_loss

    def skill_prior_loss(self, idx, obs, params, i):
        """Compute loss for skill prior."""
        if self.length_dim is not None:
            try:
                probs = functional_call(self.models['LengthPrior'],
                                        params['LengthPrior'],
                                        obs)
            except TypeError:
                pdb.set_trace()

            samples = torch.argmax(probs, dim=1)
            samples_oh = F.one_hot(samples, num_classes=len(self.hrchy))
        
            state = torch.cat([obs, samples_oh], dim=1)
            prior = functional_call(self.models['SkillPrior'],
                                    params['SkillPrior'],
                                    state)
            pdf = Normal(self.loc[idx, samples, :], self.scale[idx, samples, :])

        else:
            prior = functional_call(self.models['SkillPrior'],
                                    params['SkillPrior'],
                                    obs)

            pdf = Normal(self.loc[idx, 0, :], self.scale[idx, 0, :])

        kl_loss = kl_divergence(prior, pdf)

        if i == 0:
            wandb.log({'skill_prior/KL loss': kl_loss.mean().detach().cpu()})
            wandb.log({'skill_prior/min std': pdf.scale.min().detach().cpu()})
            wandb.log({'skill_prior/max std': pdf.scale.max().detach().cpu()})
            if kl_loss.mean() < .2:
                wandb.log({'skill_prior/KL dist':
                           wandb.Histogram(kl_loss.detach().cpu())})
            wandb.log({'skill_prior/VAE std dist':
                       wandb.Histogram(pdf.scale.detach().cpu())})
            wandb.log({'skill_prior/VAE mu dist':
                       wandb.Histogram(pdf.loc.detach().cpu())})
            
        return kl_loss.mean()
        
    def length_softmax(self, x):
        """Compute softmax for optimal length skills."""
        vecs = self.vecs.repeat(x.shape[0], 1)
        x = x.reshape(-1, 1)
        err = - torch.abs(vecs - x) / 2
        err = F.softmax(err, dim=1)

        return err
        
    def load_dataset(self):
        """Extract sequences of length max_length from episodes.

        This dataset requires to know when terminal states occur.
        """
        env = gym.make(self.env_id)
        data = env.get_dataset()

        keys = ['actions', 'observations']
        dataset = {}
        self.max_length = self.skill_length

        terminal_key = 'terminals' if 'kitchen' in self.env_id else 'timeouts'
        # 'terminals' for kitchen environment; 'timeouts' for adroit.

        terminal_idxs = np.arange(len(data[terminal_key]))
        terminal_idxs = terminal_idxs[data[terminal_key]]
        
        episode_cutoff = terminal_idxs[0] + self.max_length

        self.run = (terminal_idxs[0],
                    data['actions'][:self.max_length * (episode_cutoff // self.max_length)])
        
        idxs = []
        base_idx = np.arange(self.max_length)
        old_idx = 0
        
        for idx in terminal_idxs:
            samples = idx - old_idx - self.max_length
            if samples < 0:
                continue
            new_idx = np.repeat(base_idx[np.newaxis, :], samples, 0)
            new_idx = new_idx + np.arange(samples)[:, np.newaxis] + old_idx
            idxs.extend(new_idx.flatten())
            old_idx = idx
            
        for key in keys:
            val_dim = data[key].shape[-1] if len(data[key].shape) > 1 else 1
            seqs = np.take(data[key], idxs, axis=0)
            seqs = seqs.reshape(-1, self.max_length, val_dim).squeeze()
            dataset[key] = seqs

        self.dataset = dataset
        self.indexes = torch.arange(self.dataset['actions'].shape[0])

    def test_vae_model(self, params, path):
        actions = torch.from_numpy(self.run[-1]).to(self.device)
        actions = actions.to(torch.float32)

        recs = {}

        for i in range(len(self.hrchy)):
            z_seq, _, _, _ = self.evaluate_encoder_hrchy(actions, params, i)
            recs[i] = self.evaluate_decoder_hrchy(z_seq, params, i)[:self.run[0], :]
            error = torch.square(actions[:self.run[0], :] - recs[i]).mean()
            print(f'MSE for level {i} is {error}')
        recs[-1] = actions[:self.run[0], :]

        fullpath = f'results/{path}/'
        if not os.path.exists(fullpath):
            os.makedirs(fullpath)
        torch.save(recs, f'{fullpath}/run.pt')
        

class Drivedata(Dataset):
    """Dataset loader."""

    def __init__(self, indexes, transform=None):
        """Dataset init."""
        self.xs = indexes

    def __getitem__(self, index):
        """Get given item."""
        return self.xs[index]

    def __len__(self):
        """Compute length."""
        return len(self.xs)
