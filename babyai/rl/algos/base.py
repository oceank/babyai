from abc import ABC, abstractmethod
import torch
import numpy

from babyai.rl.format import default_preprocess_obss
from babyai.rl.utils import DictList, ParallelEnv
from babyai.rl.utils.supervised_losses import ExtraInfoCollector

from einops import rearrange
import time
time_cost = {
    "text_generation":[],
    "desc_and_update": [],
    "generate_sentence": [],
    "generate_tokens": [],
    "decoding": [],
    "tk":[],
    "vlm_BOW": [],
    "img_reshape": [],
    "media_loc": [],
    "toGPU": [],
    "update_history": [],
    "pass_desc": [],
    "model_inference":[]
}

class BaseAlgo(ABC):
    """The base class for RL algorithms."""

    def __init__(self, envs, acmodel, num_frames_per_proc, discount, lr, gae_lambda, entropy_coef,
                 value_loss_coef, max_grad_norm, recurrence, preprocess_obss, reshape_reward, aux_info):
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

        # Initialize experience values

        shape = (self.num_frames_per_proc, self.num_procs)

        self.obs = self.env.reset()
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
        time_start_all = time.time()

        time_start = time.time()
        # save the current visual observation and the prompt of the language observation in history
        self.acmodel.update_history(preprocessed_obs)
        time_cost['update_history'].append(time.time() - time_start)

        time_start = time.time()
        # generate a text description for the current visual observation and then update the history
        desc_text_tokens, time_cost_temp = self.acmodel.generate_descs_and_update_histories()
        time_cost['desc_and_update'].append(time.time() - time_start)
        for key in time_cost_temp:
            time_cost[key].append(time_cost_temp[key])

        time_start = time.time()
        preprocessed_obs.desc = self.acmodel.pass_descriptions_to_agent().to(self.device)
        time_cost['pass_desc'].append(time.time() - time_start)

        time_cost['text_generation'].append(time.time() - time_start_all)

    def show_time_cost(self):
        num_samples = len(time_cost['model_inference'])
        num_decials = 6
        msg = f"[number of samples: {num_samples}] "
        msg += f"mi: {round(sum(time_cost['model_inference'])/num_samples, num_decials)}"
        if self.acmodel.use_vlm:
            msg += f", tg: {round(sum(time_cost['text_generation'])/num_samples, num_decials)} | "
            msg += f"uh: {round(sum(time_cost['update_history'])/num_samples, num_decials)}"
            msg += f", pd: {round(sum(time_cost['pass_desc'])/num_samples, num_decials)}"
            msg += f", du: {round(sum(time_cost['desc_and_update'])/num_samples, num_decials)} | "
            msg += f"tk: {round(sum(time_cost['tk'])/num_samples, num_decials)}"
            msg += f", vB: {round(sum(time_cost['vlm_BOW'])/num_samples, num_decials)}"
            msg += f", ir: {round(sum(time_cost['img_reshape'])/num_samples, num_decials)}"
            msg += f", ml: {round(sum(time_cost['media_loc'])/num_samples, num_decials)}"
            msg += f", tG: {round(sum(time_cost['toGPU'])/num_samples, num_decials)}"
            msg += f", gs: {round(sum(time_cost['generate_sentence'])/num_samples, num_decials)} | "
            msg += f"gt: {round(sum(time_cost['generate_tokens'])/num_samples, num_decials)}"
            msg += f", dd: {round(sum(time_cost['decoding'])/num_samples, num_decials)}"
        print(msg)

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
                #start_mi = time.time()
                model_results = self.acmodel(preprocessed_obs, self.memory * self.mask.unsqueeze(1))
                #time_cost['model_inference'].append(time.time() - start_mi)
                time_cost['model_inference'].append(model_results['time_cost'])

                dist = model_results['dist']
                value = model_results['value']
                memory = model_results['memory']
                extra_predictions = model_results['extra_predictions']

            action = dist.sample()

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

            self.log_episode_return *= self.mask
            self.log_episode_reshaped_return *= self.mask

        self.show_time_cost()

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

        self.log_done_counter = 0
        self.log_return = self.log_return[-self.num_procs:]
        self.log_reshaped_return = self.log_reshaped_return[-self.num_procs:]
        self.log_num_frames = self.log_num_frames[-self.num_procs:]

        return exps, log

    @abstractmethod
    def update_parameters(self):
        pass
