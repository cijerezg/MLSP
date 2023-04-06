"""Train all models."""

from offline.offline_train import HIVES
from utilities.utils import params_extraction, load_pretrained_models
from utilities.optimization import set_optimizers
from rl.agent import MLSP
from rl.sampler import Sampler, ReplayBuffer
from datetime import datetime
from models.nns import Critic, LengthPolicy, SkillPolicy
import wandb
import os
import torch
import argparse
import pdb

os.environ['WANDB_SILENT'] = "true"

wandb.login()


def main(config=None):
    """Train all modules."""
    with wandb.init(project=f'MLSP-camera-ready-{config["env_id"]}', config=config):

        config = wandb.config

        path = config.foldername
        hives = HIVES(config)
        
        if not config.train_rl:
            if config.train_VAE or config.train_priors:
                hives.dataset_loader()

        if hives.length_dim is not None:
            length_policy = LengthPolicy(
                hives.state_dim, hives.length_dim).to(hives.device)
        else:
            length_policy = None

        skill_policy = SkillPolicy(hives.state_dim, hives.length_dim,
                                   hives.action_range, latent_dim=hives.lat_dim).to(hives.device)

        critic = Critic(hives.state_dim, hives.lat_dim, hives.length_dim).to(hives.device)
        
        sampler = Sampler(skill_policy, length_policy, hives.evaluate_decoder_hrchy, config)
        
        experience_buffer = ReplayBuffer(hives.buffer_size, sampler.env, len(hives.hrchy), hives.lat_dim)
               
        mlsp = MLSP(sampler,
                    experience_buffer,
                    hives,
                    skill_policy,
                    length_policy,
                    critic,
                    config)
        
        hives_models = list(hives.models.values())

        policy_names = ['SkillPolicy'] if mlsp.length_dim is None else ['SkillPolicy', 'LengthPolicy']
        policy_models = [mlsp.skill_policy] if mlsp.length_dim is None else [mlsp.skill_policy, mlsp.length_policy]

        models = [*hives_models, *policy_models, mlsp.critic, mlsp.critic,
                  mlsp.critic, mlsp.critic]
        names = [*hives.names, *policy_names, 'Critic1', 'Target_critic1',
                 'Critic2', 'Target_critic2']
    
        # params_path = 'multi_length_0/params_27-03-2023-12:11:47_offline.pt'
        # Case 0
        # params_path = 'case0/params_rl_02-04-2023-08:43:27_iter400000.pt'
        # Case 0 No LP
        # params_path = 'case0/params_rl_02-04-2023-10:26:24_iter400000.pt'
        params_path = config.params_path
        # params_path = 'params_27-03-2023-10:05:27_epoch500.pt'
        
        pretrained_params = load_pretrained_models(config, params_path, hives.hrchy)
        pretrained_params.extend([None] * (len(names) - len(pretrained_params)))
        
        params = params_extraction(models, names, pretrained_params)
        test_freq = config.epochs // 4
        test_freq = test_freq if test_freq > 0 else 1
    
        keys_optims = [f'Level{i}' for i in hives.hrchy]

        if mlsp.length_dim is not None:
            keys_optims.extend(['LengthPrior'])
            keys_optims.extend(['LengthPolicy'])

        keys_optims.extend(['SkillPrior', 'SkillPolicy'])
        keys_optims.extend(['Critic1', 'Critic2'])
        
        optimizers = set_optimizers(params, keys_optims, config.lr)
    
        if config.train_VAE:
            for j in range(len(hives.hrchy)):
                for e in range(config.epochs):
                    params = hives.train_vae_level(params,
                                                   optimizers,
                                                   config.beta,
                                                   j)
                    if j == len(hives.hrchy) - 1 and e % test_freq == 0:
                        print(f'Epoch is {e}')
                print('Level terminated.')
            folder = 'VAE_models'
            dt_string = datetime.now().strftime("%d-%m-%Y-%H:%M:%S")
            fullpath = f'{path}/{folder}'
            if not os.path.exists(fullpath):
                os.makedirs(fullpath)
            torch.save(params, f'{path}/{folder}/params_{dt_string}_offline.pt')
                
        if config.train_priors:
            hives.set_skill_lookup(len(hives.hrchy), params)
            
            if hives.length_dim is not None:
                for i in range(config.epochs):
                    params = hives.train_prior(params, optimizers)
                
            for i in range(config.epochs):
                params = hives.train_prior(params, optimizers,
                                           length=False)

            if config.single_length is not None:
                folder = f'single_length_{config.single_length}'
            else:
                folder = f'multi_length_{config.case}'
            dt_string = datetime.now().strftime("%d-%m-%Y-%H:%M:%S")
            fullpath = f'{path}/{folder}'
            if not os.path.exists(fullpath):
                os.makedirs(fullpath)
            torch.save(params, f'{path}/{folder}/params_{dt_string}_offline.pt')

        CASE_FOLDER = f'case{config.case}' if config.single_length is None else f'Single{config.single_length}'

        if config.train_rl:
            params = mlsp.training(params, optimizers, path, CASE_FOLDER)

        if config.render_results:
            mlsp.render_results(params, f'videos/{config.env_id}/{CASE_FOLDER}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_id', type=str, default='relocate-cloned-v1')
    parser.add_argument('--foldername', type=str, default='checkpoints')
    
    parser.add_argument('--train_VAE', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--train_priors', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--train_rl', action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument('--load_VAE_models', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--load_prior_models', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--load_rl_models', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--params_path', type=str, default='checkpoints/relocate/params_relocate.pt')

    parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--alpha_lr', type=float, default=3e-4, help='Learning rate for alpha updates')
    
    parser.add_argument('--use_SAC', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--case', type=int, default=0, help='Lengths to use')
    parser.add_argument('--single-length', type=int, default=None, help='Which single length to use')
    parser.add_argument('--render_results', action=argparse.BooleanOptionalAction, default=False)

    parser.add_argument('--offline_batch_size', type=int, default=1024, help='Batch size for VAE and priors')
    parser.add_argument('--epochs', type=int, default=501, help='Training epochs for offline models')
    parser.add_argument('--beta', type=float, default=0.2, help='Regularization parameters for ELBO loss')

    parser.add_argument('--online_batch_size', type=int, default=256, help='Batch size for online models')
    parser.add_argument('--action_range', type=int, default=4, help='action range for the skill policy')
    parser.add_argument('--discount', type=float, default=0.97, help='RL discount factor')
    parser.add_argument('--delta_skill', type=float, default=32, help='Target divergence for skills')
    parser.add_argument('--delta_length', type=float, default=24, help='Target divergence for lengths')
    parser.add_argument('--max_iterations', type=int, default=4e5, help='Max number of gradient updates')
    parser.add_argument('--buffer_size', type=int, default=4e5, help='Size of experience replay buffer')
    parser.add_argument('--critic_warmup', type=int, default=512, help='Number of iterations before training policies')
    parser.add_argument('--test_freq', type=int, default=100000, help='Frequency to save models')
    args = parser.parse_args()

    if torch.cuda.is_available():
        device = 'cuda'
        print('Using GPU')
    else:
        device = 'cpu'

    config = vars(args)
    config['device'] = device

    main(config=config)
