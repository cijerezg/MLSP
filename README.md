# Multi-length Skills with Priors for Reinforcement Learning

This repository is the official implementation of Multi-Length Skills with Prior for Reinforcement Learning.  

Neural Information Processing Systems / NeurIPS 2023

> We present MLSP, an approach to transfer multi-length skills from offline datasets to accelerate downstream learning of unseen tasks. We propose HIMES, 
a deep latent variable model that learns an embedding space of multi-length skills, and two priors, one over lengths, whose notion of optimal length is 
based on the reconstruction error of skills, and the other one over skills. We then extend RL algorithms to the multi-length case, which entails 
incorporating HIMES, the length prior, the skill prior learned from offline datasets, and a length policy for downstream learning. During inference, MLSP
first selects a length following the length policy, and then a skill of that length following the skill policy.

## Requirements

```setup
pip install -r requirements.txt
```

## Training and Evaluation

To train the all models from scratch with default hyperparameters

> Training all models at once can have very large RAM memory consumption, especially for large offline datasets. For this reason, we recommend first to train the VAE model, then the priors, and finally the RL models.

```main
python main.py --env_id ENV_ID --foldername FOLDERNAME
```

There are other possible options for training. Those are given by the following boolean arguments

- `--train_VAE` This argument determines whether to learn the embedding space for the skills.
     > Because all models use the same embedding space, it is recommended the VAE models are learned once,
	 and then are loaded for all subsequent experiments. 
- `--train_priors` This argument determines whether to learn the length and skill priors.
- `--train_rl` This argument determines whether to train the rl models (length policy, skill policy, and critics).
- `--load_VAE_models` If no VAE models are trained, then this must be set to `True`.
- `--load_prior_models` This argument determines whether to load pretrained priors models.
- `--load_rl_models` This argument determines whether to load pretrained rl models.
- `--use_SAC` If set to `True`, then the prior models are ignored, and the algorithm executed is SAC with multi-length skills.

There are other 3 important arguments. Those are:

- `--case` This set what lengths are to be used. The value required is 0 which runs 3 lengths: 4, 16, and 64. To train the models with other lengths, we refer the reader to the TODO
- `--single_length` If `None`, then the MLSP is run. If a value is passed, then SPiRL is implemented with that given length.
     > Note that the length passed here has to be a valid length, that is, it must be 4, 16, or 64.
- `--render_results` If set to `True`, then the program will create a video with the policy being executed. Additionally, it saves the resulting frame after the skill execution.

As another example

```main
python main.py --env_id kitchen-mixed-v0 --no-train_VAE --no-train_priors --no-train_rl --load_VAE_models --load_priors_models --load_rl_models --params_path checkpoints/kitchen/params_kitchen.pt --render_results
```

This command would render a video and skill frames for the kitchen environment. All arguments have a default value, which can be seen in `main.py`. If the intended value for an argument is the same as the default, then there is no need to pass the argument, e.g., `--case` has default value `0`, which was not pass because that is the intended value.


For all other hyperparmeters we refer the readers to the Further Details section.


## Pretrained Models
The pretained models are part of the repository. They can be found in the `checkpoints` folder. They are automatically loaded when the load arguments are set to `True`. If you wish to use the pretained VAE model, and priors, but train the rl models from scratch, then set `--load_VAE_models` and `--load_prior_models` to `True`, and `--load_rl_models` to `False`.

> All the models are saved in a dictionary. Therefore, it is necessary to specify which models to load.


## Results

Many metrics, including the reward, are uploaded to wandb (weight and biases). The average reward across four runs for each of the four environment is:

| Kitchen manipulation | Robot hand: relocate | Robot hand: door | Robot hand: hammer|
| ------------------ |---------------- | -------------- | ----- |
| 2.98   |     4.45         |      11.87       |  16.45 |

## Further details about the code

Additional hyperparemeters that can be modified are:

- `--lr` Learning rate for all models
- `--alpha_lr` Learning rate for updating log alpha

### Offline hyperparameters
- `--offline_batch_size` Batch size to use for offline models.
- `--epochs` The number of epochs. Applicable to the offline models.
- `--beta` Regularization parameter for the ELBO loss of the VAE model.

### Online hyperparameters
- `--online_batch_size` Batch size for gradient updates.
- `--action_range` Sets a range for the skill policy, i.e., the output is within this range. If 4, then the range is [-4, 4].
- `--discount` Discount factor used in RL.
- `--delta_length` Value for target divergence for the lengths.
- `--delta_skill` Value for target divergence for the skills.
- `--max_iterations` Max number of gradient updates (not environments steps).
- `--buffer_size` Size of the buffer containing the replay experience.
- `--critic_warmup` Number of gradient updates before the policies start being trained.
- `--checkpoint_freq` Frequency with which models are saved during RL training.


This implementation can be easily used with other environments in the D4RL library simply by changing `--env_id`. To use it with external environments, the offline dataset must be provided and loaded, which requires minor modifications to the code. 

### Technical overview of the implementation

There are four modules to our implementation.

- `models` In this module, the neural network architectures are defined.
- `utilities` As the name suggests, it contain various useful functions. The most notable one is that the hyperparameters are passed to the `hyper_arams` class inside `utils.py`. To try different lengths, go to the function `vae_hrchy_config` and modify it there directly.
- `offline` This module sets the training of the offline models.
- `rl` This module sets the training of online models. It contains two scripts. `sampler.py`, which runs the policies on the environment and prepares the data, and `agent.py` which uses the data to compute the losses and apply the gradients to update the online models.


One key difference of our implementation from other implementations of RL algorithms is that we use [functional_call](https://pytorch.org/docs/stable/generated/torch.func.functional_call.html#torch.func.functional_call) to evaluate the neural networks. The functional call requires to pass two inputs: the value to be mapped (aka the traditional input) and the weights of the neural network. This is why we store the parameters of all models in a dictionary and query the appropriate model parameters when needed.
