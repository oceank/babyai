from abc import ABC, abstractmethod
import torch
import numpy as np
import blosc

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
                 agent, num_episodes, use_subgoal_desc, use_FiLM):
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
        self.use_subgoal_desc = use_subgoal_desc
        self.use_FiLM=use_FiLM

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

            if self.use_FiLM:
                memory = torch.zeros(num_envs, self.acmodel.memory_size, device=self.device)
     
            done = [False]
            episode_num_subgoals = 0
            self.log_episode_num_frames[ep_idx] = 0
            while not done[0]:
                episode_num_subgoals += 1

                # Currently only support one process. That is, self.obs only has one element, self.obs[0]
                self.obss[ep_idx].append(self.obs[0])

                if self.use_FiLM:
                    preprocessed_obs = self.preprocess_obss(self.obs, device=self.device)

                    with torch.no_grad():
                        model_results = self.acmodel(preprocessed_obs, memory)
                        dist = model_results['dist']
                        value = model_results['value']
                        memory = model_results['memory']
                        extra_predictions = model_results['extra_predictions']

                    action = dist.sample()
                    log_probs = dist.log_prob(action)
                else: # use vlm, Flamingo
                    with torch.no_grad():
                        # pass all observed up to now in the episode i
                        model_results = self.acmodel(self.obss[ep_idx], record_subgoal_time_step=True, use_subgoal_desc=self.use_subgoal_desc)
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
                    raw_log_prob = dist.log_prob(raw_action)
                    log_probs = raw_log_prob[range(num_envs), input_ids_len-1]

                    # (b=1, ): the indice of the last token of the recent subgoal description
                    action = raw_action[range(num_envs), input_ids_len-1]
                    value = raw_value[range(num_envs), input_ids_len-1]

                # Add the description of the selected subgoal
                if self.use_subgoal_desc:
                    for obs_, env_, subgoal_idx in zip(self.obs, self.env.envs, action):
                        obs_['subgoal'] = env_.sub_goals[subgoal_idx]['desc']

                obs, reward, done, subgoals_consumed_steps = self.agent.apply_skill_batch(
                    num_envs, self.obs, self.env, action.cpu().numpy())

                self.log_episode_num_frames[ep_idx] += subgoals_consumed_steps[0]

                # Update experiences values
                self.obs = obs

                self.log_probs[ep_idx].append(log_probs)
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


class BaseAlgoFlamingoHRLIL(ABC):
    """The base class for RL algorithms."""

    def __init__(self, envs, acmodel, discount, lr, gae_lambda, entropy_coef,
                 value_loss_coef, max_grad_norm, preprocess_obss, reshape_reward,
                 agent, num_episodes, expert_model):
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

        self.expert_model = expert_model

        # Store helpers values
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_procs = len(envs) # 1

        # Initialize parameters for having the agent use subgoals
        self.agent = agent

        # Initialize experience values
        self.obs = self.env.reset()
        self.agent.reset_goal_and_subgoals(self.env.envs)

        self.obss = [None]*(self.num_episodes)
        self.rewards = [None]*(self.num_episodes)
        self.expert_actions = [None]*(self.num_episodes)
        self.expert_values  = [None]*(self.num_episodes)
        #self.agent_logits = [None]*(self.num_episodes)

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
            self.rewards[ep_idx] = []
            self.expert_actions[ep_idx] = []
            self.expert_values[ep_idx] = []
            #self.agent_logits[ep_idx] = []

            self.obs = self.env.reset() # self.obs is a list of observations from multiple environments
            self.agent.reset_goal_and_subgoals(self.env.envs)

            done = [False]
            episode_num_subgoals = 0
            self.log_episode_num_frames[ep_idx] = 0
            expert_memory = torch.zeros(num_envs, self.expert_model.memory_size, device=self.device)
            while not done[0]:
                episode_num_subgoals += 1

                with torch.no_grad():
                    preprocessed_obs = self.preprocess_obss(self.obs, device=self.device)
                    expert_result = self.expert_model(preprocessed_obs, expert_memory)
                    expert_memory = expert_result['memory']
                    expert_dist = expert_result['dist']
                    expert_value = expert_result['value']
                
                expert_action = expert_dist.probs.argmax(dim=-1)
                self.expert_actions[ep_idx].append(expert_action)
                self.expert_values[ep_idx].append(expert_value)

                # Currently only support one process. That is, self.obs only has one element, self.obs[0]
                self.obss[ep_idx].append(self.obs[0])

                with torch.no_grad():
                    # pass all observed up to now in the episode i
                    model_results = self.acmodel(self.obss[ep_idx], record_subgoal_time_step=True)
                    # dist: (b=1, max_lang_model_input_len, num_of_actions)
                    dist = model_results['dist']
                    # value: (b=1, max_lang_model_input_len)
                    raw_value = model_results['value']
                    # logit: (b=1, max_lang_model_input_len, num_of_actions)
                    #raw_logits = model_results['logits']

                    # subgoal_indice_per_sample: list of lists
                    # batch_size (number of processes) = length of input_ids_len. (it is 1 for now)
                    # number of subgoals in each episode i = input_ids_len[0][i]. batch_size=1
                    subgoal_indice_per_sample = model_results['subgoal_indice_per_sample']
                    # input_ids_len: 1-D tensor that stores indices of the recent subgoals in each process
                    input_ids_len = model_results['input_ids_len']

                # (b=1, max_lang_model_input_len)
                raw_action = dist.sample()
                # (b=1, ): the indice of the last token of the recent subgoal description
                action = raw_action[range(num_envs), input_ids_len-1]
                #logits = raw_logits[range(num_envs), input_ids_len-1, :]

                obs, reward, done, subgoals_consumed_steps = self.agent.apply_skill_batch(
                    num_envs, self.obs, self.env, action.cpu().numpy())

                self.log_episode_num_frames[ep_idx] += subgoals_consumed_steps[0]

                # Update experiences values
                self.obs = obs

                #self.agent_logits[ep_idx].append(logits)

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


            # Update log values
            self.log_total_consumed_frames += self.log_episode_num_frames[ep_idx]
            self.log_return[ep_idx] = reward[0]
            self.log_reshaped_return[ep_idx] = self.rewards[ep_idx]
            self.log_num_frames[ep_idx] = self.log_episode_num_frames[ep_idx]
            self.log_num_subgoals[ep_idx] = episode_num_subgoals
            self.log_total_num_subgoals += episode_num_subgoals


        # Flatten the data correctly, making sure that
        # each episode's data is a continuous chunk

        exps = DictList()

        #exps.agent_logits = self.agent_logits
        exps.expert_actions = self.expert_actions
        exps.expert_values = self.expert_values
        exps.obs = self.obss

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

# use HRLAgentHistory
class BaseAlgoFlamingoHRLv1(ABC):
    """The base class for RL algorithms."""

    def __init__(self, envs, discount, lr, gae_lambda, entropy_coef,
                 value_loss_coef, max_grad_norm, preprocess_obss, reshape_reward,
                 agent, num_episodes, generate_subgoal_desc=False, demos=None):
        """
        Initializes a `BaseAlgoFlamingoHRLv1` instance.

        Parameters:
        ----------
        envs : list
            a list of environments that will be run in parallel
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

        self.discount = discount
        self.lr = lr
        self.gae_lambda = gae_lambda
        self.entropy_coef = entropy_coef
        self.value_loss_coef = value_loss_coef
        self.max_grad_norm = max_grad_norm
        self.preprocess_obss = preprocess_obss or default_preprocess_obss
        self.reshape_reward = reshape_reward

        self.num_episodes = num_episodes
        self.generate_subgoal_desc = generate_subgoal_desc


        # Store helpers values
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_procs = len(envs) # 1

        # Initialize parameters for having the agent use subgoals
        self.agent = agent

        # Initialize experience values
        #self.obs = self.env.reset()
        #self.agent.reset_goal_and_subgoals(self.env.envs)
        self.obs = None
        self.histories = [None]*(self.num_episodes)

        #self.obss = [None]*(self.num_episodes)

        self.actions = [None]*(self.num_episodes)
        self.values = [None]*(self.num_episodes)
        self.rewards = [None]*(self.num_episodes)
        self.advantages = [None]*(self.num_episodes)
        self.log_probs = [None]*(self.num_episodes)

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

        # Supervise Training of the vlm that maps a history to a high-level action in a PPO fashion
        # * Randomize the collected demos
        # * For each iteration, fetch the next batch of episodes
        # * Split the batch into a number of minibatch, say N
        # * Update the vlm N times using the minibatches
        # Each episode in self.demos is a tuple as:
        #   (mission, blosc.pack_array(np.array(images)), directions, actions, completed_subgoals, reward, seed+len(demos))
        self.demos = demos
        self.batch_start_epsode_idx_in_demos = -1
        if self.demos:
            self.batch_start_epsode_idx_in_demos = 0

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
                histor/obs, action, value, reward, advantage, log_prob, returnnn.
            history:
            *   goal: the agent's goal
            *   token_seqs: a sqeuence of partial visual observations up to now: includeing attention masks and media_locations
            *   vis_obss:a sequence of images represents the agent's observational history up to now,
                including the mission goal, each subgoal's description, status, and corresponding
                visual observations.
            *   highlevel_actions
            *   highlevel_timesteps
            *   hla_hid_indices: highlevel actions' corresponding hidden state indices
            *   subgoals_status
            *   lowlevel_actions
            *   rewards

        logs : dict
            Useful stats about the training process, including the average
            reward, policy loss, value loss, etc.

        """

        num_envs = 1

        self.log_total_consumed_frames = 0
        self.log_total_num_subgoals = 0

        for ep_idx in range(self.num_episodes):

            # Initialize logging information
            self.actions[ep_idx] = []
            self.values[ep_idx] = []
            self.rewards[ep_idx] = []
            self.advantages[ep_idx] = []
            self.log_probs[ep_idx] = []

            if self.demos: # only self.actions is useful. It has the list of labels for the supervise training
                # Fetch the episode from demos
                demo_idx = self.batch_start_epsode_idx_in_demos + ep_idx
                if demo_idx >= len(self.demos):
                    continue
                demo = self.demos[demo_idx]
                goal = demo[0]
                all_images = demo[1]
                completed_subgoals = demo[4]
                reward = demo[5]
                # transform the retrieved info
                all_images = blosc.unpack_array(all_images)
                n_observations = all_images.shape[0]
                initial_obs = {'image':all_images[0]}

                # Reset the agent
                self.obs = initial_obs
                self.agent.reset_subgoal_and_skill()
                self.agent.reset_history(goal, initial_obs)
                self.agent.current_time_step = 0
                self.agent.goal = goal
                # Walkaround for using DictList. only one history per episode for now
                self.histories[ep_idx] = [self.agent.history]

                """
                For each iteration in the while loop:
                    the starting point maps to the time step when the previous subgoal is done or the mission just starts. self.agent.history contains:
                        the goal description,
                        all previous subgoals' description and their statuses
                        all the relevant visual observations (including the ones finishing the previous subgoal or the initial observation)

                    self.agent.current_subgoal_start_time = self.agent.current_time_step
                    Check the next recorded piece in the demo until the next completed subgoal(s) are found or the mission is done
                    Update self.agent.current_time_step properly
                    Save the high-level action as the target label
                    If the mission is not done, accumulate the following to self.agent.history
                        the description of the subgoal indicated by the high-level action
                        the sequence of visual observations that completes the subgoal
                        the status of the subgoal
                """
                episode_num_subgoals = 0
                done = False
                while not done:
                    # Update current_subgoal_start_time for searching for the next completed subgoal
                    # Here, current_subgoal_start_time refers the timesteps when the mission starts (timestep 0) or the previous subgoal finishes
                    self.agent.current_subgoal_start_time = self.agent.current_time_step

                    # When the mission is None done and the the next completed subgoal has not been found, check the next timestep.
                    # In collected demonstrations, the 'completed_subgoals' info at time t+1 are saved in the indice t in the demo record.
                    found_next_completed_subgoal = False
                    while not found_next_completed_subgoal:
                        self.agent.current_time_step += 1 # the next timestep to check
                        found_next_completed_subgoal = (len(completed_subgoals[self.agent.current_time_step-1]) > 0)
                        done = (self.agent.current_time_step == n_observations)
                        if done:
                            break

                    # If not done and found a completed subgoal
                    if (not done) and found_next_completed_subgoal:
                        episode_num_subgoals += 1
                        # Set the current subgoal idx as the label for the current history
                        #   At some point, the agent may complete several DropNextTo subgoals simulataneously.
                        #   Then, randomly select one as the completed target subgoal
                        completed_subgoal_indices = completed_subgoals[self.agent.current_time_step-1]
                        highlevel_action = completed_subgoal_indices[0]
                        if len(completed_subgoal_indices) > 1:
                            highlevel_action = np.random.choice(completed_subgoal_indices)
                        self.agent.history.highlevel_actions.append(highlevel_action)
                        self.agent.history.highlevel_time_steps.append(self.agent.current_subgoal_start_time)
                        #   update the info of the current subgoal in the agent
                        self.agent.current_subgoal = self.agent.subgoal_set.all_subgoals[highlevel_action]
                        self.agent.current_subgoal_idx = self.agent.current_subgoal[0]
                        self.agent.current_subgoal_instr = self.agent.current_subgoal[1]
                        self.agent.current_subgoal_desc = self.agent.current_subgoal_instr.instr_desc
                        self.agent.current_subgoal_status = 0 # 0: in progress, 1: success, 2: faliure

                        # Accumulate input info in the history for the next subgoal prediction
                        #   append the sugboal's description to the agent's history.
                        b_idx = 0
                        start = self.agent.history.token_seq_lens[b_idx]
                        new_subgoal_token_len = self.agent.subgoals_token_lens[highlevel_action]
                        end = start + new_subgoal_token_len
                        self.agent.history.token_seqs['input_ids'][b_idx, start:end] = self.agent.subgoals_token_seqs['input_ids'][highlevel_action, :new_subgoal_token_len]
                        self.agent.history.token_seqs['attention_mask'][b_idx, start:end] = 1
                        #   append the elapsed visual observations to the agent's history
                        t = self.agent.current_subgoal_start_time + 1
                        while t <= self.agent.current_time_step :
                            #accumulate the visual observations
                            obs = {'image': all_images[t]}
                            self.agent.history.vis_obss.append(obs)
                            t += 1
                        #   append the subgoal's status to the agent's history.
                        #   update hla_hid_indices[-1] to have it refers to the index of hidden state that is used by VLM to predict the next subgoal
                        self.agent.current_subgoal_status = 1 # 0: in progress, 1: success, 2: faliure
                        self.agent.update_history_with_subgoal_status()

                if found_next_completed_subgoal: # If the mission is done and found a completed subgoal
                    episode_num_subgoals += 1
                    # Set the current subgoal idx as the label for the current history
                    completed_subgoal_indices = completed_subgoals[self.agent.current_time_step-1]
                    highlevel_action = completed_subgoal_indices[0]
                    if len(completed_subgoal_indices) > 1:
                        highlevel_action = np.random.choice(completed_subgoal_indices)
                    self.agent.history.highlevel_actions.append(highlevel_action)
                    self.agent.history.highlevel_time_steps.append(self.agent.current_subgoal_start_time)
                else:
                    # When the mission is done and no subgoal completion is recoginized,
                    # then, remove the last elem in hla_hid_indices, which is used to fetch
                    # the embedding of VLM for predicting the next subgoal. It is set inside
                    # update_history_with_subgoal_status()
                    self.agent.history.hla_hid_indices.pop()

                # Update actions (high-level actions)
                self.actions[ep_idx] = [torch.tensor(action, device=self.device) for action in self.agent.history.highlevel_actions]
                # The last frame in the episode is not saved in the demo
                self.log_episode_num_frames[ep_idx] = n_observations + 1

                # NOT NEEDED FOR SUPERISE TRAINING.
                # Set dummy values in order not to CHANGE THE PPO CODE REGARDING self.log_probs, self.values, self.rewards, and self.advantages.
                self.log_probs[ep_idx]  = [torch.tensor(0.0, device=self.device) for i in range(episode_num_subgoals)]
                self.values[ep_idx]     = [torch.tensor(0.0, device=self.device) for i in range(episode_num_subgoals)]
                self.rewards[ep_idx]    = [torch.tensor(0.0, device=self.device) for i in range(episode_num_subgoals)]
                self.advantages[ep_idx] = [torch.tensor(0.0, device=self.device) for i in range(episode_num_subgoals)]
                reshaped_reward = torch.tensor(reward, device=self.device)
                if self.reshape_reward is not None:
                    action = self.actions[ep_idx][-1]
                    reshaped_reward = torch.tensor(self.reshape_reward(obs, action, reward, done), device=self.device)
                self.rewards[ep_idx][episode_num_subgoals-1] = reshaped_reward
            else:
                # self.obs is a list of observations from multiple environments
                # Currently only support one process. That is, self.env.reset() only has one element.
                self.obs = self.env.reset()[0]
                initial_obs = self.obs
                goal = self.obs['mission']
                cur_env = self.env.envs[0]
                self.agent.on_reset(cur_env, goal, initial_obs, propose_first_subgoal=False)
                # walkaround for using DictList. only one history per episode for now
                self.histories[ep_idx] = [self.agent.history]

                done = False
                episode_num_subgoals = 0
                self.log_episode_num_frames[ep_idx] = 0
                while not done:
                    # Check if a subgoal is needed. If yes, generate a subgoal description.
                    if self.agent.need_new_subgoal():
                        with torch.no_grad():
                            highlevel_action, log_prob, value = self.agent.propose_new_subgoal(cur_env)
                            episode_num_subgoals += 1

                    # Apply the corresponding skill to solve the subgoal until it is done
                    while (not done) and self.agent.current_subgoal_status == 0:
                        # Pretrained Skills Are Fixed:
                        #   no_grad() is applied inside act() such that the pretrained
                        #   skill will not be modified during backward propagation.
                        result = self.agent.act(self.obs)
                        action = result['action'].item()
                        obs, reward, done, _ = cur_env.step(action)
                        # Update the current_time_step and accumulate information to the agent's history
                        self.agent.current_time_step += 1
                        self.agent.accumulate_env_info_to_subgoal_history(action, obs, reward, done)
                        # check if the current subgoal is done
                        self.agent.verify_current_subgoal(action)
                        if done or (self.agent.current_subgoal_status != 0): # the current subgoal is done
                            # subgoal_success = self.agent.current_subgoal_status == 1
                            # append the subgoal status to the agent's history
                            self.agent.update_history_with_subgoal_status()

                        # Update the observation that will be used by the current/new skill
                        self.obs = obs

                    # The current subgoal is done and save experiences values for the high-level policy
                    self.log_probs[ep_idx].append(log_prob)
                    self.actions[ep_idx].append(highlevel_action)
                    self.values[ep_idx].append(value)
                    ## The 'reward' is associated with the timestep when the current subgoal is done.
                    ## Reward reshaping might not be necessary for the high-level policy.
                    ## It is used when training low-level policy / predefined skill.
                    if self.reshape_reward is not None:
                        self.rewards[ep_idx].append(
                            torch.tensor(self.reshape_reward(obs, action, reward, done), device=self.device)
                        )
                    else:
                        self.rewards[ep_idx].append(torch.tensor(reward, device=self.device))
                    self.advantages[ep_idx].append(0) # initialize the advantages for the episode, ep_idx

            # The episode is done
            ## Update log values
            ### current_time_step indicates the number of steps elapsed in this episode
            self.log_episode_num_frames[ep_idx] = self.agent.current_time_step
            self.log_total_consumed_frames += self.log_episode_num_frames[ep_idx]
            self.log_return[ep_idx] = reward # a scalar value, reward from the environment
            self.log_reshaped_return[ep_idx] = self.rewards[ep_idx] # reshaped reward
            self.log_num_frames[ep_idx] = self.log_episode_num_frames[ep_idx]
            self.log_num_subgoals[ep_idx] = episode_num_subgoals
            self.log_total_num_subgoals += episode_num_subgoals


            if self.demos is None:
                ## Add advantage and return to experiences
                #   The mission is done after executing the last subgoal.
                #   So, the advantage at the time point of the last subgoal is:
                #       advantage = delta + discount * gae_lambda * 0 (next_advantage)
                #       delta = reward - value + 0 (next_value)
                #   (subgoal_time_step == (episode_num_subgoals - 1)): the last subgoal
                for subgoal_time_step in reversed(range(episode_num_subgoals)):
                    is_last_subgoal = subgoal_time_step == (episode_num_subgoals - 1)
                    next_mask = 0 if is_last_subgoal else 1
                    next_value = 0 if is_last_subgoal else self.values[ep_idx][subgoal_time_step+1]
                    next_advantage = 0 if is_last_subgoal else self.advantages[ep_idx][subgoal_time_step+1]

                    delta = self.rewards[ep_idx][subgoal_time_step] + self.discount * next_value * next_mask - self.values[ep_idx][subgoal_time_step]
                    self.advantages[ep_idx][subgoal_time_step] = delta + self.discount * self.gae_lambda * next_advantage * next_mask

        # Flatten the data correctly, making sure that
        # each episode's data is a continuous chunk

        exps = DictList()

        exps.history = self.histories
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

        # update the start index of the next batch of episodes
        if self.demos:
            self.batch_start_epsode_idx_in_demos += self.num_episodes

        return exps, log

    @abstractmethod
    def update_parameters(self):
        pass
