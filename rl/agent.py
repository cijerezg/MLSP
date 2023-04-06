"""Training RL algorithm."""

import sys
sys.path.insert(0, '../')

import torch
from utilities.optimization import Adam_update
from utilities.utils import hyper_params, process_frames
from torch.func import functional_call
import torch.nn.functional as F
import torch.nn as nn
from torch.distributions.kl import kl_divergence
import wandb
import numpy as np
from datetime import datetime
from torch.optim import Adam
import torch.autograd as autograd
import os
import pdb
from stable_baselines3.common.utils import polyak_update
import copy


MAX_SKILL_KL = 100
MAX_LENGTH_KL = 20
INIT_LOG_ALPHA = 0

class MLSP(hyper_params):
    def __init__(self,
                 sampler,
                 experience_buffer,
                 vae,
                 skill_policy,
                 length_policy,
                 critic,
                 args):

        super().__init__(args)

        self.sampler = sampler
        self.critic = critic
        self.skill_policy = skill_policy
        self.length_policy = length_policy
        self.vae = vae
        self.experience_buffer = experience_buffer

        self.log_alpha_skill = torch.tensor(INIT_LOG_ALPHA, dtype=torch.float32,
                                            requires_grad=True,
                                            device=self.device)
        self.optimizer_alpha_skill = Adam([self.log_alpha_skill], lr=args.alpha_lr)

        if self.length_dim is not None:
            self.log_alpha_length = torch.tensor(INIT_LOG_ALPHA, dtype=torch.float32,
                                                 requires_grad=True,
                                                 device=self.device)
            self.optimizer_alpha_length = Adam([self.log_alpha_length], lr=args.alpha_lr)

        self.reward_per_episode = 0
        self.total_episode_counter = 0
        self.reward_logger = [0]
        self.log_data = 0
        self.log_data_freq = 1000

    def training(self, params, optimizers, path, name):
        self.iterations = 0
        ref_params = copy.deepcopy(params)

        while self.iterations < self.max_iterations:

            critic_warmup = True if self.iterations < self.critic_warmup else False

            if self.iterations == 0:
                obs = None
                done = False

            params, obs, done = self.training_iteration(params, done,
                                                        optimizers,
                                                        ref_params,
                                                        obs=obs,
                                                        critic_warmup=critic_warmup)

            if self.iterations % self.test_freq == 0 and self.iterations > 0:
                dt_string = datetime.now().strftime("%d-%m-%Y-%H:%M:%S")
                print(f'Current iteration is {self.iterations}')
                print(dt_string)
                fullpath = f'{path}/{name}'
                if not os.path.exists(fullpath):
                    os.makedirs(fullpath)
                filename = f'{path}/{name}/params_rl_{dt_string}_iter{self.iterations}.pt'
                torch.save(params, filename)

            if self.iterations % self.log_data_freq == 0:
                wandb.log({'Iterations': self.iterations})
            self.iterations += 1

        return params

    def training_iteration(self,
                           params,
                           done,
                           optimizers,
                           ref_params,
                           obs=None,
                           critic_warmup=False):
        
        obs, data = self.sampler.skill_iteration(params, done, obs)

        next_obs, rew, z, next_z, le, next_l, done, l_soft, next_l_soft = data
        aux_l = np.zeros((le.size, len(self.hrchy)))
        aux_l[np.arange(le.size), le] = 1
        aux_next_l = np.zeros((next_l.size, len(self.hrchy)))
        aux_next_l[np.arange(next_l.size), next_l] = 1
        onehot_l = aux_l
        onehot_next_l = aux_next_l

        if self.env_key == 'kitchen':
            self.reward_per_episode += rew
        elif self.env_key == 'adroit':
            self.reward_per_episode = max(self.reward_per_episode, rew)
            
        if done:
            wandb.log({'Reward per episode': self.reward_per_episode})
            self.reward_logger.append(self.reward_per_episode)
            self.reward_per_episode = 0
            self.total_episode_counter += 1

        self.experience_buffer.add(
            obs, next_obs, z, next_z, onehot_l, onehot_next_l, rew, done)

        log_data = True if self.log_data % self.log_data_freq == 0 else False

        if len(self.reward_logger) > 25 and log_data:
            wandb.log({'Average reward over 100 eps': np.mean(self.reward_logger[-100:])}, step=self.iterations)
            wandb.log({'Total Episodes': self.total_episode_counter})

        self.log_data = (self.log_data + 1) % self.log_data_freq

        policy_names = ['SkillPolicy'] if self.length_dim is None else ['SkillPolicy', 'LengthPolicy']

        if self.experience_buffer.size > self.online_batch_size:
            policy_losses, critic1_loss, critic2_loss = self.losses(params, log_data)           
            losses = [critic1_loss, critic2_loss] if critic_warmup else [*policy_losses, critic1_loss, critic2_loss]
            names = ['Critic1', 'Critic2'] if critic_warmup else [*policy_names, 'Critic1', 'Critic2']
            params = Adam_update(params, losses, names, optimizers)
            polyak_update(params['Critic1'].values(),
                          params['Target_critic1'].values(), 0.005)
            polyak_update(params['Critic2'].values(),
                          params['Target_critic2'].values(), 0.005)
            if log_data:
                with torch.no_grad():
                    dist_init1 = self.distance_to_params(params, ref_params, 'Critic1', 'Critic1')
                    dist_init2 = self.distance_to_params(params, ref_params, 'Critic2', 'Critic2')
                    dist_init_pol = self.distance_to_params(params, ref_params, 'SkillPolicy', 'SkillPolicy')
                    if self.length_dim is not None:
                        dist_init_len = self.distance_to_params(params, ref_params, 'LengthPolicy', 'LengthPolicy')
                    
                wandb.log({'Critic/Distance to init weights 1': dist_init1,
                           'Critic/Distance to init weights 2': dist_init2,
                           'Policy/Distance to init weights Skills': dist_init_pol})

                if self.length_dim is not None:
                    wandb.log({'Policy/Distance to init weights Length': dist_init_len})
           
        return params, next_obs, done

    def losses(self, params, log_data):
        batch = self.experience_buffer.sample(batch_size=self.online_batch_size)

        obs = torch.from_numpy(batch.observations).to(self.device)
        next_obs = torch.from_numpy(batch.next_observations).to(self.device)
        z = torch.from_numpy(batch.z).to(self.device)
        next_z = torch.from_numpy(batch.next_z).to(self.device)
        length = torch.from_numpy(batch.l).to(self.device)
        next_l = torch.from_numpy(batch.next_l).to(self.device)
        rew = torch.from_numpy(batch.rewards).to(self.device)

        target_critic_arg = torch.cat([next_obs, next_z, next_l], dim=1)
        z_prior_arg = torch.cat([obs, length], dim=1)

        with torch.no_grad():
            if self.length_dim is not None:
                l_prior = self.eval_length_prior(obs, params)
                                
            z_prior = self.eval_skill_prior(z_prior_arg, params)
            
            q_target1, q_target2 = self.eval_critic(target_critic_arg, params,
                                                    target_critic=True)

        q_target = torch.cat((q_target1, q_target2), dim=1)
        q_target, _ = torch.min(q_target, dim=1)
            
        critic_arg = torch.cat([obs, z, length], dim=1)

        q1, q2 = self.eval_critic(critic_arg, params)
        
        if log_data:
            if self.length_dim is not None:
                wandb.log({'Priors/Length prior std': l_prior.std().detach().cpu()})
            with torch.no_grad():
                dist1 = self.distance_to_params(params, params, 'Critic1', 'Target_critic1')
                dist2 = self.distance_to_params(params, params, 'Critic2', 'Target_critic2')
                wandb.log({'Critic/Distance critic to target 1': dist1,
                           'Critic/Distance critic to target 2': dist2})
            wandb.log({'Critic/Target nn vals unclamped':
                       wandb.Histogram(q_target.cpu())})

        exponents = self.discount_exponents(length)

        discount = torch.pow(self.discount, exponents)

        q_target = rew + (discount * q_target).reshape(-1, 1)
        q_target = torch.clamp(q_target, min=-100, max=100)

        critic1_loss = F.mse_loss(q1.squeeze(), q_target.squeeze(),
                                  reduction='none')
        critic2_loss = F.mse_loss(q2.squeeze(), q_target.squeeze(),
                                  reduction='none')

        if log_data:
            wandb.log(
                {'Critic/Target vals clamped': wandb.Histogram(q_target.cpu()),
                 'Critic/Error dist 1': wandb.Histogram(critic1_loss.detach().cpu()),
                 'Critic/Error dist 2': wandb.Histogram(critic2_loss.detach().cpu()),
                 'Critic/High error percentages 1': sum(critic1_loss > 1) / critic1_loss.shape[0],
                 'Critic/High error percentages 2': sum(critic2_loss > 1) / critic2_loss.shape[0],
                 'Critic/Median error 1': torch.median(critic1_loss).detach().cpu(),
                 'Critic/Median error 2': torch.median(critic2_loss).detach().cpu(),
                 'Critic/Max error 1': critic1_loss.max().detach().cpu(),
                 'Critic/Max error 2': critic2_loss.max().detach().cpu()})

            rew_thrh = 0.0

            if rew.max() > rew_thrh:
                wandb.log(
                    {'Critic/nonzero_reward_error 1': critic1_loss[rew.squeeze() > rew_thrh].mean().detach().cpu(),
                     'Critic/nonzero_reward_error 2': critic2_loss[rew.squeeze() > rew_thrh].mean().detach().cpu(),
                     'Critic/Q1 positive rewards': wandb.Histogram(q1[rew.squeeze() > rew_thrh].detach().cpu()),
                     'Critic/Q2 positive rewards': wandb.Histogram(q2[rew.squeeze() > rew_thrh].detach().cpu()),
                     'Critic/Target positive rewards': wandb.Histogram(q_target[rew.squeeze() > rew_thrh].detach().cpu()),
                     'Critic/Nonzero rewards': wandb.Histogram(rew[rew > rew_thrh].detach().cpu())})
                        
        critic1_loss = critic1_loss.mean()
        critic2_loss = critic2_loss.mean()
        
        if log_data:
            wandb.log(
                {'Critic/Critic1 Grad Norm': self.get_gradient(critic1_loss, params, 'Critic1'),
                 'Critic/Critic2 Grad Norm': self.get_gradient(critic2_loss, params, 'Critic2'),
                 'Critic/critic1_loss': critic1_loss,
                 'Critic/critic2_loss': critic2_loss})

        if self.length_dim is not None:
            l_soft, l_sample = self.eval_length_policy(obs, params)
            l_soft_exp = torch.exp(l_soft)
            samples_oh = F.one_hot(l_sample, num_classes=len(self.hrchy))
            
            state = torch.cat([obs, samples_oh], dim=1)
            
            z_sample, pdf, mu, std = self.eval_skill_policy(state, params)
            q_pi_arg_l = torch.cat([obs, z_sample.detach(), l_soft_exp], dim=1)
            q_pi_arg = torch.cat([obs, z_sample, l_soft_exp.detach()], dim=1)

        else:
            z_sample, pdf, mu, std = self.eval_skill_policy(obs, params)
            q_pi_arg = torch.cat([obs, z_sample], dim=1)

        q_pi1, q_pi2 = self.eval_critic(q_pi_arg, params)
        q_pi = torch.cat((q_pi1, q_pi2), dim=1)
        q_pi, _ = torch.min(q_pi, dim=1)

        if self.length_dim is not None:
            q_pi1_l, q_pi2_l = self.eval_critic(q_pi_arg_l, params)
            q_pi_l = torch.cat((q_pi1_l, q_pi2_l), dim=1)
            q_pi_l, _ = torch.min(q_pi_l, dim=1)

        if self.use_SAC:
            skill_prior = torch.clamp(pdf.entropy(), max=MAX_SKILL_KL).mean()
        else:
            skill_prior = torch.clamp(kl_divergence(pdf, z_prior), max=MAX_SKILL_KL).mean()

        if self.length_dim is not None:
            if self.use_SAC:
                length_prior = self.SAC_length_KL(l_soft).mean()
            else:
                length_prior = F.kl_div(l_soft, l_prior, log_target=True,
                                        reduction='batchmean')
                
            alpha_length = torch.exp(self.log_alpha_length).detach()
            length_prior_loss = alpha_length * length_prior
        
        alpha_skill = torch.exp(self.log_alpha_skill).detach()
        skill_prior_loss = alpha_skill * skill_prior

        q_val_policy = -torch.mean(q_pi)
        if self.length_dim is not None:
            q_val_policy_l = -torch.mean(q_pi_l)

        if log_data:
            if self.length_dim is not None:
                wandb.log({'Policy/Length Prior Grad Norm':
                           self.get_gradient(length_prior_loss, params, 'LengthPolicy')})
                wandb.log({'Policy/Qcritic Length Grad Norm':
                           self.get_gradient(q_val_policy_l, params, 'LengthPolicy')})

            wandb.log(
                {'Policy/Skill Prior Grad Norm': self.get_gradient(skill_prior_loss, params, 'SkillPolicy'),
                 'Policy/Qcritic Skill Grad Norm': self.get_gradient(q_val_policy, params, 'SkillPolicy')})

        skill_policy_loss = q_val_policy + skill_prior_loss
            
        if self.length_dim is not None:
            length_policy_loss = q_val_policy_l + length_prior_loss
            policy_losses = [skill_policy_loss, length_policy_loss]
        else:
            policy_losses = [skill_policy_loss]

        loss_alpha_skill = torch.exp(self.log_alpha_skill) * \
            (self.delta_skill - skill_prior).detach()

        self.optimizer_alpha_skill.zero_grad()
        loss_alpha_skill.backward()
        self.optimizer_alpha_skill.step()

        if self.length_dim is not None:
            loss_alpha_length = torch.exp(self.log_alpha_length) * \
                (self.delta_length - length_prior).detach()
            self.optimizer_alpha_length.zero_grad()
            loss_alpha_length.backward()
            self.optimizer_alpha_length.step()
            
        if log_data:
            wandb.log(
                {'Policy/current_q_values': wandb.Histogram(q_pi.detach().cpu()),
                 'Policy/current_q_values_average': q_pi.detach().mean().cpu(),
                 'Policy/current_q_values_max': q_pi.detach().max().cpu(),
                 'Policy/current_q_values_min': q_pi.detach().min().cpu(),
                 'Policy/Z abs value mean': z_sample.abs().mean().detach().cpu(),
                 'Policy/Z std': z_sample.std().detach().cpu(),
                 'Policy/Z distribution': wandb.Histogram(z_sample.detach().cpu()),
                 'Policy/Mean STD': std.mean().detach().cpu(),
                 'Policy/Mu dist': wandb.Histogram(mu.detach().cpu())})
            if self.length_dim is not None:
                wandb.log(
                    {'Priors/Alpha Length': alpha_length.detach().cpu(),
                     'Priors/length_prior_loss': length_prior.detach().cpu()})
                for i in range(l_soft.shape[1]):
                    wandb.log({f'Policy/Length means {i}': torch.exp(l_soft[:, i]).mean()})

            wandb.log(
                {'Priors/Alpha skill': alpha_skill.detach().cpu(),
                 'Priors/skill_prior_loss': skill_prior.detach().cpu()})

            wandb.log(
                {'Critic/critic1_loss': critic1_loss,
                 'Critic/critic2_loss': critic2_loss,
                 'Critic/Q1': wandb.Histogram(q1.detach().cpu()),
                 'Critic/Q2': wandb.Histogram(q2.detach().cpu()),
                 'Critic/Target values w reward': wandb.Histogram(q_target.cpu())})

            wandb.log({'Reward Percentage': sum(rew > 0.0) / self.online_batch_size})
        
        return policy_losses, critic1_loss, critic2_loss

    def eval_length_prior(self, obs, params):
        l_prior = functional_call(self.vae.models['LengthPrior'],
                                  params['LengthPrior'], obs)
        return l_prior

    def eval_length_policy(self, obs, params):
        l_soft, l_sample = functional_call(self.length_policy,
                                           params['LengthPolicy'], obs)
        return l_soft, l_sample

    def eval_skill_prior(self, state, params):
        z_prior = functional_call(self.vae.models['SkillPrior'],
                                  params['SkillPrior'], state)
        return z_prior

    def eval_skill_policy(self, state, params):
        sample, pdf, mu, std = functional_call(self.skill_policy,
                                               params['SkillPolicy'],
                                               state)
        return sample, pdf, mu, std

    def eval_critic(self, arg, params, target_critic=False):
        if target_critic:
            name1, name2 = 'Target_critic1', 'Target_critic2'
        else:
            name1, name2 = 'Critic1', 'Critic2'

        q1 = functional_call(self.critic, params[name1], arg)
        q2 = functional_call(self.critic, params[name2], arg)

        return q1, q2

    def discount_exponents(self, length):
        aux_exp = torch.argmax(length, dim=1)
        exp = torch.zeros_like(aux_exp)
        for lev in self.level_lengths:
            exp[aux_exp == lev] = self.level_lengths[lev] / self.level_lengths[0]

        return exp
            
    def SAC_length_KL(self, l_pol):
        lengths = l_pol.shape[1]
        uniform_length_prior = torch.tensor([1 / lengths] * lengths,
                                            dtype=torch.float32).to(self.device)
        log_uni_length_prior = torch.log(uniform_length_prior)
        rew_kl_len = F.kl_div(l_pol, log_uni_length_prior,
                              reduction='none', log_target=True).mean(1)

        return rew_kl_len
        

    def get_gradient(self, x, params, key):
        grads = autograd.grad(x, params[key].values(), retain_graph=True,
                              allow_unused=True)

        grads = [grad for grad in grads if grad is not None]
        try:
            grads_vec = nn.utils.parameters_to_vector(grads)
        except RuntimeError:
            pdb.set_trace()
        return torch.norm(grads_vec).detach().cpu()

    def distance_to_params(self, params1, params2, name1, name2):
        with torch.no_grad():
            vec1 = nn.utils.parameters_to_vector(params1[name1].values())
            target_vec1 = nn.utils.parameters_to_vector(params2[name2].values())
        return torch.norm(vec1 - target_vec1)

    def render_results(self, params, foldername):
        test_episodes = 1
        
        for j in range(test_episodes):
            done = False
            obs = None

            frames = []
            self.sampler.env.reset()

            while not done:
                obs, done, frames = self.sampler.skill_iteration_with_frames(params,
                                                                             done=done,
                                                                             obs=obs,
                                                                             frames=frames)

            process_frames(frames, self.env_id, f'{foldername}/test_{j}')
