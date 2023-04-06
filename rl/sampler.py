"""Create sampler for RL with buffer."""

import sys
sys.path.insert(0, '../')

from utilities.utils import hyper_params, AttrDict
import gym
import d4rl
import numpy as np
from torch.func import functional_call
import torch
import torch.nn.functional as F


WIDTH = 4 * 640
HEIGHT = 4 * 480

class Sampler(hyper_params):
    def __init__(self, skill_policy, length_policy, decoder, args):
        super().__init__(args)

        self.length_policy = length_policy
        self.skill_policy = skill_policy
        self.decoder = decoder
        MAX_EPISODE_STEPS = 384 if self.env_key == 'adroit' else 384

        self.env = gym.make(self.env_id)
        self.env._max_episode_steps = MAX_EPISODE_STEPS

    def skill_execution(self, actions, frames=None):
        obs_trj, rew_trj, done_trj = [], [], []
        aux_frames = []
        
        for i in range(actions.shape[0]):
            next_obs, rew, done, info = self.env.step(actions[i, :])
            if frames is not None:
                if self.env_key != 'kitchen':
                    frame = self.env.sim.render(width=WIDTH, height=HEIGHT,
                                                mode='offscreen',
                                                camera_name='vil_camera')
                    aux_frames.append(frame)
                else:
                    frame = self.env.sim.render(width=WIDTH, height=HEIGHT)
                    aux_frames.append(frame)
                    
            if self.env_key != 'kitchen':
                done = info['goal_achieved'] if len(info) == 1 else True
            obs_trj.append(next_obs)
            rew_trj.append(rew)
            done_trj.append(done)
        if frames is not None:
            frames.append(aux_frames)

        return obs_trj, rew_trj, done_trj, frames

    def skill_step(self, params, obs, frames=None):
        obs_t = torch.from_numpy(obs).to(self.device).to(torch.float32)
        obs_t = obs_t.reshape(1, -1)

        with torch.no_grad():
            if self.length_dim is not None:
                l_soft, l_sample = functional_call(self.length_policy,
                                                   params['LengthPolicy'],
                                                   obs_t)
                
                samples_oh = F.one_hot(l_sample, num_classes=len(self.hrchy))
 
                state = torch.cat([obs_t, samples_oh], dim=1)

                z_sample, _, _, _ = functional_call(self.skill_policy,
                                                    params['SkillPolicy'],
                                                    state)

            else:
                l_sample = torch.tensor((0,))
                z_sample, _, _, _ = functional_call(self.skill_policy,
                                                    params['SkillPolicy'],
                                                    obs_t)

            actions = self.decoder(z_sample, params, l_sample)
            res_lev = self.level_lengths[l_sample.cpu().numpy()[0]]
            actions = actions.reshape(res_lev, actions.shape[-1])

        actions = actions.cpu().detach().numpy()
        clipped_actions = np.clip(actions, -1, 1)
            
        obs_trj, rew_trj, done_trj, frames = self.skill_execution(clipped_actions,
                                                                  frames=frames)

        if frames is not None:
            done = True if sum(done_trj) > 0 else False
            return obs_trj[-1], done, frames

        next_obs_t = torch.from_numpy(obs_trj[-1]).to(self.device).to(torch.float32)
        next_obs_t = next_obs_t.reshape(1, -1)

        with torch.no_grad():
            if self.length_dim is not None:
                next_l_soft, next_l_sample = functional_call(self.length_policy,
                                                             params['LengthPolicy'],
                                                             next_obs_t)

                next_samples_oh = F.one_hot(next_l_sample, num_classes=len(self.hrchy))

                state = torch.cat([next_obs_t, next_samples_oh], dim=1)

                next_z_sample, _, _, _ = functional_call(self.skill_policy,
                                                         params['SkillPolicy'],
                                                         state)

            else:
                next_l_sample = torch.tensor(0)
                next_z_sample, _, _, _ = functional_call(self.skill_policy,
                                                         params['SkillPolicy'],
                                                         next_obs_t)
                l_soft = torch.tensor(0)
                next_l_soft = torch.tensor(0)
           
        next_obs = obs_trj[-1]
        if self.env_key == 'kitchen':
            rew = sum(rew_trj)
        elif self.env_key == 'adroit':
            rew = rew_trj[-1]

        z = z_sample.cpu().numpy()
        l_samp = l_sample.cpu().numpy()
        next_z = next_z_sample.cpu().numpy()
        next_l = next_l_sample.cpu().numpy()
        done = True if sum(done_trj) > 0 else False
        l_soft = l_soft.cpu().numpy()
        next_l_soft = next_l_soft.cpu().numpy()

        return next_obs, rew, z, next_z, l_samp, next_l, done, l_soft, next_l_soft

    def skill_iteration(self, params, done=False, obs=None):
        if done or obs is None:
            obs = self.env.reset()

        return obs, self.skill_step(params, obs)

    def skill_iteration_with_frames(self, params, done=False, obs=None, frames=None):
        if done or obs is None:
            obs = self.env.reset()

        frames = self.skill_step(params, obs, frames)

        return frames
    
   
class ReplayBuffer:
    def __init__(self, size, env, lengths, lat_dim):
        self.obs_buf = np.zeros((size, *env.observation_space.shape), dtype=np.float32)
        self.next_obs_buf = np.zeros((size, *env.observation_space.shape), dtype=np.float32)
        self.z_buf = np.zeros((size, lat_dim), dtype=np.float32)
        self.next_z_buf = np.zeros((size, lat_dim), dtype=np.float32)
        self.l_buf = np.zeros((size, lengths), dtype=np.float32)
        self.next_l_buf = np.zeros((size, lengths), dtype=np.float32)
        self.rew_buf = np.zeros((size, 1), dtype=np.float32)
        self.done_buf = np.zeros((size, 1), dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size

    def add(self, obs, next_obs, z, next_z, l, next_l, rew, done):
        self.obs_buf[self.ptr] = obs
        self.next_obs_buf[self.ptr] = next_obs
        self.z_buf[self.ptr] = z
        self.next_z_buf[self.ptr] = next_z
        self.l_buf[self.ptr] = l
        self.next_l_buf[self.ptr] = next_l
        self.rew_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size)
        batch = AttrDict(observations=self.obs_buf[idxs],
                         next_observations=self.next_obs_buf[idxs],
                         z=self.z_buf[idxs],
                         next_z=self.next_z_buf[idxs],
                         l=self.l_buf[idxs],
                         next_l=self.next_l_buf[idxs],
                         rewards=self.rew_buf[idxs],
                         dones=self.done_buf[idxs])
        return batch        
