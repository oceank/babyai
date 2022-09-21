from abc import ABC, abstractmethod
import torch
import numpy as np

from babyai.rl.format import default_preprocess_obss
from babyai.rl.utils import DictList, ParallelEnv
from babyai.rl.utils.supervised_losses import ExtraInfoCollector

from einops import rearrange

class BaseAlgo(ABC):
    """The base class for RL algorithms."""

    def __init__(self, envs, acmodel, num_frames_per_proc, discount, lr, gae_lambda, entropy_coef,
                 value_loss_coef, max_grad_norm, recurrence, preprocess_obss, reshape_reward, aux_info,
                 use_subgoal=False, agent=None):
        """
        Initializes a `BaseAlgo` instance.

        Parameters:
        ----------
        envs : list
            a list of environments that will be run in parallel
        acmodel : torch.Module
            the model
        num_frames_per_proc : int
            the number of frames collected by every process for an update
        discount : float
            the discount for future rewards
        lr : float
            the learning rate for optimizers
        gae_lambda : float
            the lambda coefficient in the GAE formula
            ([Schulman et al., 2015](https://arxiv.org/abs/1506.02438))
        entropy_coef : float
            the weight of the entropy cost in the final objective
        value_loss_coef : float
            the weight of the value loss in the final objective
        max_grad_norm : float
            gradient will be clipped to be at most this value
        recurrence : int
            the number of steps the gradient is propagated back in time
        preprocess_obss : function
            a function that takes observations returned by the environment
            and converts them into the format that the model can handle
        reshape_reward : function
            a function that shapes the reward, takes an
            (observation, action, reward, done) tuple as an input
        aux_info : list
            a list of strings corresponding to the name of the extra information
            retrieved from the environment for supervised auxiliary losses

        """
        # Store parameters

        self.env = ParallelEnv(envs)
        self.acmodel = acmodel
        self.acmodel.train()
        self.num_frames_per_proc = num_frames_per_proc
        self.discount = discount
        self.lr = lr
        self.gae_lambda = gae_lambda
        self.entropy_coef = entropy_coef
        self.value_loss_coef = value_loss_coef
        self.max_grad_norm = max_grad_norm
        self.recurrence = recurrence
        self.preprocess_obss = preprocess_obss or default_preprocess_obss
        self.reshape_reward = reshape_reward
        self.aux_info = aux_info

        # Store helpers values

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_procs = len(envs)
        self.num_frames = self.num_frames_per_proc * self.num_procs


        assert self.num_frames_per_proc % self.recurrence == 0

        # Initialize parameters for having the agent use subgoals 
        self.use_subgoal = use_subgoal
        self.agent = agent

        # Initialize experience values

        shape = (self.num_frames_per_proc, self.num_procs)

        self.obs = self.env.reset()

        if self.use_subgoal:
            assert agent is not None
            agent.reset_goal_and_subgoals(self.env.envs)

            self.log_episode_num_primitive_steps = torch.zeros(self.num_procs, device=self.device)
            self.log_num_primitive_steps = [0] * self.num_procs
            self.log_total_num_primitive_steps = 0 # across all parallel envs per model update

        self.obss = [None]*(shape[0])

        self.memory = torch.zeros(shape[1], self.acmodel.memory_size, device=self.device)
        self.memories = torch.zeros(*shape, self.acmodel.memory_size, device=self.device)

        self.mask = torch.ones(shape[1], device=self.device)
        self.masks = torch.zeros(*shape, device=self.device)
        self.actions = torch.zeros(*shape, device=self.device, dtype=torch.int)
        self.values = torch.zeros(*shape, device=self.device)
        self.rewards = torch.zeros(*shape, device=self.device)
        self.advantages = torch.zeros(*shape, device=self.device)
        self.log_probs = torch.zeros(*shape, device=self.device)

        if self.aux_info:
            self.aux_info_collector = ExtraInfoCollector(self.aux_info, shape, self.device)

        # Initialize log values

        self.log_episode_return = torch.zeros(self.num_procs, device=self.device)
        self.log_episode_reshaped_return = torch.zeros(self.num_procs, device=self.device)
        self.log_episode_num_frames = torch.zeros(self.num_procs, device=self.device)

        self.log_done_counter = 0
        self.log_return = [0] * self.num_procs
        self.log_reshaped_return = [0] * self.num_procs
        self.log_num_frames = [0] * self.num_procs

    # Add 'desc' element to the input dictionary, preprocessed_obs
    def add_desc_to_obs(self, preprocessed_obs):
        # save the current visual observation and the prompt of the language observation in history
        self.acmodel.update_history(preprocessed_obs)

        # generate a text description for the current visual observation and then update the history
        desc_text_tokens = self.acmodel.generate_descs_and_update_histories()

        preprocessed_obs.desc = self.acmodel.pass_descriptions_to_agent().to(self.device)

    def collect_experiences(self):
        """Collects rollouts and computes advantages.

        Runs several environments concurrently. The next actions are computed
        in a batch mode for all environments at the same time. The rollouts
        and advantages from all environments are concatenated together.

        Returns
        -------
        exps : DictList
            Contains actions, rewards, advantages etc as attributes.
            Each attribute, e.g. `exps.reward` has a shape
            (self.num_frames_per_proc * num_envs, ...). k-th block
            of consecutive `self.num_frames_per_proc` frames contains
            data obtained from the k-th environment. Be careful not to mix
            data from different environments!
        logs : dict
            Useful stats about the training process, including the average
            reward, policy loss, value loss, etc.

        """

        # The list has self.num_frames_per_proc elements
        # Each elmenent is a preprocessed_obs object
        # preprocessed_obs.image: (batch_size, image_dim)
        # preprocessed_obs.instr: (batch_size, instr_dim)
        # preprocessed_obs.desc : (batch_size, max_length)
        #preprocessed_obs_all= []

        if self.use_subgoal:
            self.log_total_num_primitive_steps = 0
        
        for i in range(self.num_frames_per_proc):
            # Do one agent-environment interaction

            preprocessed_obs = self.preprocess_obss(self.obs, device=self.device)

            if self.acmodel.use_vlm:
                # Store the goals in the header of the list, history[0]
                if len(self.acmodel.history) == 0:
                    self.acmodel.initialize_history_with_goals(self.obs)
                
                self.add_desc_to_obs(preprocessed_obs)
            
                #preprocessed_obs_all.append(preprocessed_obs)
                self.obss[i] = preprocessed_obs
            else:
                self.obss[i] = self.obs

            with torch.no_grad():
                model_results = self.acmodel(preprocessed_obs, self.memory * self.mask.unsqueeze(1))
                dist = model_results['dist']
                value = model_results['value']
                memory = model_results['memory']
                extra_predictions = model_results['extra_predictions']

            action = dist.sample()

            if self.use_subgoal:
                obs, reward, done, subgoals_consumed_steps = self.agent.apply_skill_batch(self.num_procs, self.obs, self.env, action.cpu().numpy())
                # FixMe: currently use the subgoals stored in the environment instance to properly
                #        setup the instruction texts for subgoals before querying the policy model.
                #        In future, the subgoals are generated by the agent and will not be available
                #        from a environment instance.
                
                subgoals_consumed_steps = torch.tensor(subgoals_consumed_steps, device=self.device)
                self.log_total_num_primitive_steps += subgoals_consumed_steps.sum().item()
                self.log_episode_num_primitive_steps += subgoals_consumed_steps
            else:
                obs, reward, done, env_info = self.env.step(action.cpu().numpy())

            if self.aux_info:
                env_info = self.aux_info_collector.process(env_info)
                # env_info = self.process_aux_info(env_info)

            # Update experiences values

            #self.obss[i] = self.obs
            self.obs = obs

            self.memories[i] = self.memory
            self.memory = memory

            self.masks[i] = self.mask
            self.mask = 1 - torch.tensor(done, device=self.device, dtype=torch.float)
            self.actions[i] = action
            self.values[i] = value
            if self.reshape_reward is not None:
                self.rewards[i] = torch.tensor([
                    self.reshape_reward(obs_, action_, reward_, done_)
                    for obs_, action_, reward_, done_ in zip(obs, action, reward, done)
                ], device=self.device)
            else:
                self.rewards[i] = torch.tensor(reward, device=self.device)
            self.log_probs[i] = dist.log_prob(action)

            if self.aux_info:
                self.aux_info_collector.fill_dictionaries(i, env_info, extra_predictions)

            # Update log values

            self.log_episode_return += torch.tensor(reward, device=self.device, dtype=torch.float)
            self.log_episode_reshaped_return += self.rewards[i]
            self.log_episode_num_frames += torch.ones(self.num_procs, device=self.device)

            for i, done_ in enumerate(done):
                if done_:
                    self.log_done_counter += 1
                    self.log_return.append(self.log_episode_return[i].item())
                    self.log_reshaped_return.append(self.log_episode_reshaped_return[i].item())
                    self.log_num_frames.append(self.log_episode_num_frames[i].item())

                    if self.use_subgoal:
                        self.log_num_primitive_steps.append(self.log_episode_num_primitive_steps[i].item())

            self.log_episode_return *= self.mask
            self.log_episode_reshaped_return *= self.mask
            self.log_episode_num_frames *= self.mask

            if self.use_subgoal:
                self.log_episode_num_primitive_steps *= self.mask

        # Add advantage and return to experiences

        preprocessed_obs = self.preprocess_obss(self.obs, device=self.device)
        if self.acmodel.use_vlm:
            self.add_desc_to_obs(preprocessed_obs)

        with torch.no_grad():
            next_value = self.acmodel(preprocessed_obs, self.memory * self.mask.unsqueeze(1))['value']

        for i in reversed(range(self.num_frames_per_proc)):
            next_mask = self.masks[i+1] if i < self.num_frames_per_proc - 1 else self.mask
            next_value = self.values[i+1] if i < self.num_frames_per_proc - 1 else next_value
            next_advantage = self.advantages[i+1] if i < self.num_frames_per_proc - 1 else 0

            delta = self.rewards[i] + self.discount * next_value * next_mask - self.values[i]
            self.advantages[i] = delta + self.discount * self.gae_lambda * next_advantage * next_mask

        # Flatten the data correctly, making sure that
        # each episode's data is a continuous chunk

        exps = DictList()

        if self.acmodel.use_vlm: # all preprocessed_obs are stored in self.obss list
            exps.obs = DictList()
            exps.obs.image = self.obss[0].image[None, :]
            exps.obs.instr = self.obss[0].instr[None, :]
            exps.obs.desc  = self.obss[0].desc[None , :]
            for preprocessed_obs in self.obss[1:]: # concatenate along the time dim
                exps.obs.image = torch.cat([exps.obs.image, preprocessed_obs.image[None, :]], dim=0)
                exps.obs.instr = torch.cat([exps.obs.instr, preprocessed_obs.instr[None, :]], dim=0)
                exps.obs.desc  = torch.cat([exps.obs.desc, preprocessed_obs.desc[None, :]]  , dim=0)
            exps.obs.image = rearrange(exps.obs.image, 't b h w c -> (b t) h w c')
            exps.obs.instr = rearrange(exps.obs.instr, 't b l -> (b t) l')
            exps.obs.desc  = rearrange(exps.obs.desc, 't b l -> (b t) l')
        else: # all raw obs are stored in self.obss
            exps.obs = [
                self.obss[i][j]
                for j in range(self.num_procs)
                for i in range(self.num_frames_per_proc)
                ]
            # Preprocess experiences
            exps.obs = self.preprocess_obss(exps.obs, device=self.device)
 
        # In commments below T is self.num_frames_per_proc, P is self.num_procs,
        # D is the dimensionality

        # T x P x D -> P x T x D -> (P * T) x D
        exps.memory = self.memories.transpose(0, 1).reshape(-1, *self.memories.shape[2:])
        # T x P -> P x T -> (P * T) x 1
        exps.mask = self.masks.transpose(0, 1).reshape(-1).unsqueeze(1)

        # for all tensors below, T x P -> P x T -> P * T
        exps.action = self.actions.transpose(0, 1).reshape(-1)
        exps.value = self.values.transpose(0, 1).reshape(-1)
        exps.reward = self.rewards.transpose(0, 1).reshape(-1)
        exps.advantage = self.advantages.transpose(0, 1).reshape(-1)
        exps.returnn = exps.value + exps.advantage
        exps.log_prob = self.log_probs.transpose(0, 1).reshape(-1)

        if self.aux_info:
            exps = self.aux_info_collector.end_collection(exps)

        # Log some values

        keep = max(self.log_done_counter, self.num_procs)

        log = {
            "return_per_episode": self.log_return[-keep:],
            "reshaped_return_per_episode": self.log_reshaped_return[-keep:],
            "num_frames_per_episode": self.log_num_frames[-keep:],
            "num_frames": self.num_frames,
            "episodes_done": self.log_done_counter,
        }
        if self.use_subgoal:
            log["num_frames_per_episode"] = self.log_num_primitive_steps[-keep:]
            log["num_frames"] = self.log_total_num_primitive_steps

            log["num_high_level_actions_per_episode"] = self.log_num_frames[-keep:]
            log["num_high_level_actions"] = self.num_frames

            self.log_num_primitive_steps = self.log_num_primitive_steps[-self.num_procs:]
        else:
            log["num_frames_per_episode"] = self.log_num_frames[-keep:]
            log["num_frames"] = self.num_frames

        self.log_done_counter = 0
        self.log_return = self.log_return[-self.num_procs:]
        self.log_reshaped_return = self.log_reshaped_return[-self.num_procs:]
        self.log_num_frames = self.log_num_frames[-self.num_procs:]


        return exps, log

    @abstractmethod
    def update_parameters(self):
        pass


class BaseAlgoFlamingoHRL(ABC):
    """The base class for RL algorithms."""

    def __init__(self, envs, acmodel, discount, lr, gae_lambda, entropy_coef,
                 value_loss_coef, max_grad_norm, preprocess_obss, reshape_reward,
                 agent, num_episodes):
        """
        Initializes a `BaseAlgo` instance.

        Parameters:
        ----------
        envs : list
            a list of environments that will be run in parallel
        acmodel : torch.Module
            the model
        num_frames_per_proc : int
            the number of frames collected by every process for an update
        discount : float
            the discount for future rewards
        lr : float
            the learning rate for optimizers
        gae_lambda : float
            the lambda coefficient in the GAE formula
            ([Schulman et al., 2015](https://arxiv.org/abs/1506.02438))
        entropy_coef : float
            the weight of the entropy cost in the final objective
        value_loss_coef : float
            the weight of the value loss in the final objective
        max_grad_norm : float
            gradient will be clipped to be at most this value
        recurrence : int
            the number of steps the gradient is propagated back in time
        preprocess_obss : function
            a function that takes observations returned by the environment
            and converts them into the format that the model can handle
        reshape_reward : function
            a function that shapes the reward, takes an
            (observation, action, reward, done) tuple as an input
        aux_info : list
            a list of strings corresponding to the name of the extra information
            retrieved from the environment for supervised auxiliary losses

        """
        # Store parameters

        self.env = ParallelEnv(envs)
        self.acmodel = acmodel
        self.acmodel.train()

        self.discount = discount
        self.lr = lr
        self.gae_lambda = gae_lambda
        self.entropy_coef = entropy_coef
        self.value_loss_coef = value_loss_coef
        self.max_grad_norm = max_grad_norm
        self.preprocess_obss = preprocess_obss or default_preprocess_obss
        self.reshape_reward = reshape_reward

        self.num_episodes = num_episodes

        # Store helpers values

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_procs = len(envs) # 1

        #self.num_frames = self.num_frames_per_proc * self.num_procs

        #assert self.num_frames_per_proc % self.recurrence == 0

        # Initialize parameters for having the agent use subgoals 
        self.agent = agent

        # Initialize experience values
        #shape = (self.num_frames_per_proc, self.num_procs)

        self.obs = self.env.reset()

        self.agent.reset_goal_and_subgoals(self.env.envs)

        #self.log_episode_num_primitive_steps = torch.zeros(self.num_procs, device=self.device)
        #self.log_num_primitive_steps = [0] * self.num_procs
        #self.log_total_num_primitive_steps = 0 # across all parallel envs per model update

        self.obss = [None]*(self.num_episodes)
        #self.mask = torch.ones(shape[1], device=self.device)
        #self.masks = torch.zeros(*shape, device=self.device)

        self.actions = [None]*(self.num_episodes)
        self.values = [None]*(self.num_episodes)
        self.rewards = [None]*(self.num_episodes)
        self.advantages = [None]*(self.num_episodes)
        self.log_probs = [None]*(self.num_episodes)

        '''
        self.mask = torch.ones(shape[1], device=self.device)
        self.masks = torch.zeros(*shape, device=self.device)
        self.actions = torch.zeros(*shape, device=self.device, dtype=torch.int)
        self.values = torch.zeros(*shape, device=self.device)
        self.rewards = torch.zeros(*shape, device=self.device)
        self.advantages = torch.zeros(*shape, device=self.device)
        self.log_probs = torch.zeros(*shape, device=self.device)
        '''

        # Initialize log values

        self.log_episode_return = torch.zeros(self.num_episodes, device=self.device)
        self.log_episode_reshaped_return = torch.zeros(self.num_episodes, device=self.device)

        self.log_episode_num_frames = torch.zeros(self.num_episodes, device=self.device, dtype=torch.int)
        self.log_total_consumed_frames = 0
        self.log_total_num_subgoals = 0
        self.log_done_counter = 0
        self.log_return = [0] * self.num_episodes
        self.log_reshaped_return = [0] * self.num_episodes
        self.log_num_frames = [0] * self.num_episodes
        self.log_num_subgoals = [0] * self.num_episodes


    def collect_experiences(self):
        """Collects rollouts and computes advantages.

        Runs several environments concurrently. The next actions are computed
        in a batch mode for all environments at the same time. The rollouts
        and advantages from all environments are concatenated together.

        Returns
        -------
        exps : DictList
            Contains actions, rewards, advantages etc as attributes.
            Each attribute, e.g. `exps.reward` is a list with a length of
            self.num_episode and each of its element is a list represents
            the reward at a time step. Here, the time step corresponds to
            the high-level policy in the HRL. That is, it is the time point
            when the corresponding subgoal is done.
            The full list of attributes is:
                obs, action, value, reward, advantage, log_prob, returnnn.
        logs : dict
            Useful stats about the training process, including the average
            reward, policy loss, value loss, etc.

        """

        num_envs = 1

        self.log_total_consumed_frames = 0
        self.log_total_num_subgoals = 0
        
        for ep_idx in range(self.num_episodes):
            self.obss[ep_idx] = []
            self.actions[ep_idx] = []
            self.values[ep_idx] = []
            self.rewards[ep_idx] = []
            self.advantages[ep_idx] = []
            self.log_probs[ep_idx] = []

            self.obs = self.env.reset() # self.obs is a list of observations from multiple environments
            self.agent.reset_goal_and_subgoals(self.env.envs)

            done = [False]
            episode_num_subgoals = 0
            self.log_episode_num_frames[ep_idx] = 0
            while not done[0]:
                episode_num_subgoals += 1

                # Currently only support one process. That is, self.obs only has one element, self.obs[0]
                self.obss[ep_idx].append(self.obs[0])

                with torch.no_grad():
                    # pass all observed up to now in the episode i
                    model_results = self.acmodel(self.obss[ep_idx], record_subgoal_time_step=True)
                    # dist: (b=1, max_lang_model_input_len, num_of_actions)
                    dist = model_results['dist']
                    # value: (b=1, max_lang_model_input_len)
                    raw_value = model_results['value']
                    # subgoal_indice_per_sample: list of lists
                    # batch_size (number of processes) = length of input_ids_len. (it is 1 for now)
                    # number of subgoals in each episode i = input_ids_len[0][i]. batch_size=1
                    subgoal_indice_per_sample = model_results['subgoal_indice_per_sample']
                    # input_ids_len: 1-D tensor that stores indices of the recent subgoals in each process
                    input_ids_len = model_results['input_ids_len']

                # (b=1, max_lang_model_input_len)
                raw_action = dist.sample()
                # (b=1, ): the indice of the last token of the recent subgoal description
                action = raw_action[range(num_envs), input_ids_len]
                value = raw_value[range(num_envs), input_ids_len]

                obs, reward, done, subgoals_consumed_steps = self.agent.apply_skill_batch(
                    num_envs, self.obs, self.env, action.cpu().numpy())

                self.log_episode_num_frames[ep_idx] += subgoals_consumed_steps[0]


                # Update experiences values
                self.obs = obs

                #self.masks[i] = self.mask
                #self.mask = 1 - torch.tensor(done, device=self.device, dtype=torch.float)

                self.actions[ep_idx].append(action)
                self.values[ep_idx].append(value)
                if self.reshape_reward is not None:
                    self.rewards[ep_idx].append(
                        torch.tensor([
                            self.reshape_reward(obs_, action_, reward_, done_)
                            for obs_, action_, reward_, done_ in zip(obs, action, reward, done)
                            ], device=self.device
                        )
                    )
                else:
                    self.rewards[ep_idx].append(torch.tensor(reward, device=self.device))
                #self.log_probs[i] = dist.log_prob(action)
                raw_log_prob = dist.log_prob(raw_action)
                self.log_probs[ep_idx].append(raw_log_prob[range(num_envs), input_ids_len])

                self.advantages[ep_idx].append(0) # initialize the advantages for the episode, ep_idx


            # Update log values
            self.log_total_consumed_frames += self.log_episode_num_frames[ep_idx]
            self.log_return[ep_idx] = reward[0]
            self.log_reshaped_return[ep_idx] = self.rewards[ep_idx]
            self.log_num_frames[ep_idx] = self.log_episode_num_frames[ep_idx]
            self.log_num_subgoals[ep_idx] = episode_num_subgoals
            self.log_total_num_subgoals += episode_num_subgoals


            # Add advantage and return to experiences

            # When the mission is done after executing the last subgoal. So, the advantage at the time point of
            # the last subgoal is: delta = reward - value
            for subgoal_time_step in reversed(range(episode_num_subgoals)):
                next_mask = 1 if subgoal_time_step < episode_num_subgoals - 1 else 0
                next_value = self.values[ep_idx][subgoal_time_step+1] if subgoal_time_step < episode_num_subgoals - 1 else 0
                next_advantage = self.advantages[ep_idx][subgoal_time_step+1] if subgoal_time_step < episode_num_subgoals - 1 else 0

                delta = self.rewards[ep_idx][subgoal_time_step] + self.discount * next_value * next_mask - self.values[ep_idx][subgoal_time_step]
                self.advantages[ep_idx][subgoal_time_step] = delta + self.discount * self.gae_lambda * next_advantage * next_mask

        # Flatten the data correctly, making sure that
        # each episode's data is a continuous chunk

        exps = DictList()
 
        exps.obs = self.obss
        exps.action = self.actions
        exps.value = self.values
        exps.reward = self.rewards
        exps.advantage = self.advantages
        exps.log_prob = self.log_probs
        exps.returnn = [
            [value+adv for value, adv in zip(values_eps, adv_eps)]
            for values_eps, adv_eps in zip(exps.value, exps.advantage)
            ]

        # Log some values

        log = {
            "return_per_episode": self.log_return,
            "reshaped_return_per_episode": [[x.item() for x in y] for y in self.log_reshaped_return],
            "num_frames_per_episode": [x.item() for x in self.log_num_frames],
            "num_frames": self.log_total_consumed_frames.item(),
            "episodes_done": self.num_episodes,
            "num_high_level_actions_per_episode": self.log_num_subgoals,
            "num_high_level_actions": self.log_total_num_subgoals
        }

        return exps, log

    @abstractmethod
    def update_parameters(self):
        pass
