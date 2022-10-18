import numpy
import torch
import torch.nn.functional as F


from babyai.rl.algos.base import BaseAlgo, BaseAlgoFlamingoHRL, BaseAlgoFlamingoHRLIL


class PPOAlgo(BaseAlgo):
    """The class for the Proximal Policy Optimization algorithm
    ([Schulman et al., 2015](https://arxiv.org/abs/1707.06347))."""

    def __init__(self, envs, acmodel, num_frames_per_proc=None, discount=0.99, lr=7e-4, beta1=0.9, beta2=0.999,
                 gae_lambda=0.95,
                 entropy_coef=0.01, value_loss_coef=0.5, max_grad_norm=0.5, recurrence=4,
                 adam_eps=1e-5, clip_eps=0.2, epochs=4, batch_size=256, preprocess_obss=None,
                 reshape_reward=None, aux_info=None, use_subgoal=False, agent=None, randomize_subbatch=True):
        num_frames_per_proc = num_frames_per_proc or 128

        super().__init__(envs, acmodel, num_frames_per_proc, discount, lr, gae_lambda, entropy_coef,
                         value_loss_coef, max_grad_norm, recurrence, preprocess_obss, reshape_reward,
                         aux_info, use_subgoal=use_subgoal, agent=agent)

        self.clip_eps = clip_eps
        self.epochs = epochs
        self.batch_size = batch_size

        assert self.batch_size % self.recurrence == 0

        self.optimizer = torch.optim.Adam(self.acmodel.parameters(), lr, (beta1, beta2), eps=adam_eps)
        self.batch_num = 0

        self.randomize_subbatch = randomize_subbatch

    def update_parameters(self):
        # Collect experiences

        exps, logs = self.collect_experiences()
        '''
        exps is a DictList with the following keys ['obs', 'memory', 'mask', 'action', 'value', 'reward',
         'advantage', 'returnn', 'log_prob'] and ['collected_info', 'extra_predictions'] if we use aux_info
        exps.obs is a DictList with the following keys ['image', 'instr']
        exps.obj.image is a (n_procs * n_frames_per_proc) x image_size 4D tensor
        exps.obs.instr is a (n_procs * n_frames_per_proc) x (max number of words in an instruction) 2D tensor
        exps.memory is a (n_procs * n_frames_per_proc) x (memory_size = 2*image_embedding_size) 2D tensor
        exps.mask is (n_procs * n_frames_per_proc) x 1 2D tensor
        if we use aux_info: exps.collected_info and exps.extra_predictions are DictLists with keys
        being the added information. They are either (n_procs * n_frames_per_proc) 1D tensors or
        (n_procs * n_frames_per_proc) x k 2D tensors where k is the number of classes for multiclass classification
        '''

        for _ in range(self.epochs):
            # Initialize log values

            log_entropies = []
            log_values = []
            log_policy_losses = []
            log_value_losses = []
            log_grad_norms = []

            log_losses = []

            '''
            For each epoch, we create int(total_frames / batch_size + 1) batches, each of size batch_size (except
            maybe the last one. Each batch is divided into sub-batches of size recurrence (frames are contiguous in
            a sub-batch), but the position of each sub-batch in a batch and the position of each batch in the whole
            list of frames is random thanks to self._get_batches_starting_indexes().
            '''

            for inds in self._get_batches_starting_indexes():
                # inds is a numpy array of indices that correspond to the beginning of a sub-batch
                # there are as many inds as there are batches
                # Initialize batch values

                batch_entropy = 0
                batch_value = 0
                batch_policy_loss = 0
                batch_value_loss = 0
                batch_loss = 0

                # Initialize memory

                memory = exps.memory[inds]

                for i in range(self.recurrence):
                    # Create a sub-batch of experience
                    sb = exps[inds + i]

                    # Compute loss

                    model_results = self.acmodel(sb.obs, memory * sb.mask)
                    dist = model_results['dist']
                    value = model_results['value']
                    memory = model_results['memory']
                    extra_predictions = model_results['extra_predictions']

                    entropy = dist.entropy().mean()

                    ratio = torch.exp(dist.log_prob(sb.action) - sb.log_prob)
                    surr1 = ratio * sb.advantage
                    surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * sb.advantage
                    policy_loss = -torch.min(surr1, surr2).mean()

                    value_clipped = sb.value + torch.clamp(value - sb.value, -self.clip_eps, self.clip_eps)
                    surr1 = (value - sb.returnn).pow(2)
                    surr2 = (value_clipped - sb.returnn).pow(2)
                    value_loss = torch.max(surr1, surr2).mean()

                    loss = policy_loss - self.entropy_coef * entropy + self.value_loss_coef * value_loss

                    # Update batch values

                    batch_entropy += entropy.item()
                    batch_value += value.mean().item()
                    batch_policy_loss += policy_loss.item()
                    batch_value_loss += value_loss.item()
                    batch_loss += loss

                    # Update memories for next epoch

                    if i < self.recurrence - 1:
                        exps.memory[inds + i + 1] = memory.detach()

                # Update batch values

                batch_entropy /= self.recurrence
                batch_value /= self.recurrence
                batch_policy_loss /= self.recurrence
                batch_value_loss /= self.recurrence
                batch_loss /= self.recurrence

                # Update actor-critic

                self.optimizer.zero_grad()
                batch_loss.backward()
                grad_norm = sum(p.grad.data.norm(2) ** 2 for p in self.acmodel.parameters() if p.grad is not None) ** 0.5
                torch.nn.utils.clip_grad_norm_(self.acmodel.parameters(), self.max_grad_norm)
                self.optimizer.step()

                # Update log values

                log_entropies.append(batch_entropy)
                log_values.append(batch_value)
                log_policy_losses.append(batch_policy_loss)
                log_value_losses.append(batch_value_loss)
                log_grad_norms.append(grad_norm.item())
                log_losses.append(batch_loss.item())

        # Log some values

        logs["entropy"] = numpy.mean(log_entropies)
        logs["value"] = numpy.mean(log_values)
        logs["policy_loss"] = numpy.mean(log_policy_losses)
        logs["value_loss"] = numpy.mean(log_value_losses)
        logs["grad_norm"] = numpy.mean(log_grad_norms)
        logs["loss"] = numpy.mean(log_losses)

        return logs

    def _get_batches_starting_indexes(self):
        """Gives, for each batch, the indexes of the observations given to
        the model and the experiences used to compute the loss at first.
        Returns
        -------
        batches_starting_indexes : list of list of int
            the indexes of the experiences to be used at first for each batch

        """

        indexes = numpy.arange(0, self.num_frames, self.recurrence)
        if self.randomize_subbatch:
            indexes = numpy.random.permutation(indexes)

        num_indexes = self.batch_size // self.recurrence
        batches_starting_indexes = [indexes[i:i + num_indexes] for i in range(0, len(indexes), num_indexes)]

        return batches_starting_indexes


class PPOAlgoFlamingoHRL(BaseAlgoFlamingoHRL):
    """The class for the Proximal Policy Optimization algorithm
    ([Schulman et al., 2015](https://arxiv.org/abs/1707.06347))."""

    def __init__(self, envs, acmodel, discount=0.99, lr=7e-4, beta1=0.9, beta2=0.999,
                 gae_lambda=0.95,
                 entropy_coef=0.01, value_loss_coef=0.5, max_grad_norm=0.5,
                 adam_eps=1e-5, clip_eps=0.2, epochs=4, preprocess_obss=None,
                 reshape_reward=None, agent=None, num_episodes=None, use_subgoal_desc=False,
                 num_episodes_per_batch=1, use_FiLM=False, average_loss_by_subgoals=True, episode_weight_type=0):
        num_episodes = num_episodes or 10

        super().__init__(envs, acmodel, discount, lr, gae_lambda, entropy_coef,
                         value_loss_coef, max_grad_norm, preprocess_obss, reshape_reward,
                         agent=agent, num_episodes=num_episodes, use_subgoal_desc=use_subgoal_desc, use_FiLM=use_FiLM)

        self.clip_eps = clip_eps
        self.epochs = epochs

        self.optimizer = torch.optim.Adam(self.acmodel.parameters(), lr, (beta1, beta2), eps=adam_eps)

        self.num_episodes_per_batch=num_episodes_per_batch
        self.average_loss_by_subgoals=average_loss_by_subgoals
        self.episode_weight_type=episode_weight_type

    def update_parameters(self):
        # Collect experiences

        exps, logs = self.collect_experiences()
        '''
        exps is a DictList with the following keys:
            'obs', 'action', 'value', 'reward', 'advantage', 'returnn', 'log_prob'.
        Each attribute, e.g. `exps.reward` is a list with a length of self.num_episode and each of its
        element is a list represents the reward at a time step. Here, the time step corresponds to the
        high-level policy in the HRL. That is, it is the time point when the corresponding subgoal is done.

        '''

        if self.average_loss_by_subgoals:
            logs = self.update_parameters_parallel_episodes(exps, logs)
            return logs
    

        num_envs = 1 # num of processes
        num_of_subgoals_per_episode = [len(actions) for actions in exps.action]

        for _ in range(self.epochs):
            # Initialize log values

            log_entropies = []
            log_values = []
            log_policy_losses = []
            log_value_losses = []
            log_grad_norms = []

            log_losses = []

            # for each epoch, we create self.num_episodes batches. One episode maps to one batch.
            episode_ids = numpy.arange(0, self.num_episodes)
            episode_ids = numpy.random.permutation(episode_ids)
            num_batches = self.num_episodes//self.num_episodes_per_batch
            for batch_group_idx in range(num_batches):
                batch_entropy = 0
                batch_value = 0
                batch_policy_loss = 0
                batch_value_loss = 0
                batch_loss = 0
                batch_num_of_subgoals = 0

                batch_start       = batch_group_idx*self.num_episodes_per_batch
                batch_after_end   = (batch_group_idx+1)*self.num_episodes_per_batch
                for ep_idx in episode_ids[batch_start:batch_after_end]:

                    num_of_subgoals = num_of_subgoals_per_episode[ep_idx]
                    # Create an episode of experience
                    ep = exps[ep_idx]
                    value_subgoals = torch.zeros(num_envs, num_of_subgoals, device=self.device)
                    entropy_subgoals = torch.zeros(num_envs, num_of_subgoals, device=self.device)
                    log_prob_subgoals = torch.zeros(num_envs, num_of_subgoals, device=self.device)

                    if self.use_FiLM:
                        memory = torch.zeros(num_envs, self.acmodel.memory_size, device=self.device)
                        for i in range(num_of_subgoals):
                            preprocessed_obs = self.preprocess_obss([ep.obs[i]], device=self.device)
                            model_results = self.acmodel(preprocessed_obs, memory)
                            dist = model_results['dist']
                            value = model_results['value']
                            memory = model_results['memory']
                            entropy = dist.entropy().mean()

                            entropy_subgoals[0, i]  = entropy
                            value_subgoals[0, i]    = value[0]
                            log_prob_subgoals[0, i] = dist.log_prob(ep.action[i])
                    else: # use Flamingo model
                        model_results = self.acmodel(ep.obs, record_subgoal_time_step=True)
                        dist = model_results['dist']
                        raw_value = model_results['value']
                        # subgoal_indice_per_sample: list of lists
                        # batch_size (number of processes) = length of input_ids_len. (it is 1 for now)
                        # number of subgoals in each episode i = input_ids_len[0][i]. batch_size=1
                        subgoal_indice_per_sample = model_results['subgoal_indice_per_sample']
                        # input_ids_len: 1-D tensor that stores indices of the recent subgoals in each process
                        input_ids_len = model_results['input_ids_len']

                        raw_entropy = dist.entropy()

                        # currently support one process/one environment
                        value_subgoals[range(num_envs), :] = raw_value[range(num_envs), subgoal_indice_per_sample[0]]
                        entropy_subgoals[range(num_envs), :] = raw_entropy[range(num_envs), subgoal_indice_per_sample[0]]
                        for i in range(num_of_subgoals):
                            log_prob_subgoals[0, i] = dist.log_prob(ep.action[i])[0, subgoal_indice_per_sample[0, i]]

                    for i in range(num_of_subgoals):
                        value = value_subgoals[0, i]
                        entropy = entropy_subgoals[0, i]
                        # Compute loss
                        #entropy = dist.entropy().mean()

                        ratio = torch.exp(log_prob_subgoals[0, i] - ep.log_prob[i])
                        surr1 = ratio * ep.advantage[i]
                        surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * ep.advantage[i]
                        policy_loss = -torch.min(surr1, surr2).mean()

                        value_clipped = ep.value[i] + torch.clamp(value - ep.value[i], -self.clip_eps, self.clip_eps)
                        surr1 = (value - ep.returnn[i]).pow(2)
                        surr2 = (value_clipped - ep.returnn[i]).pow(2)
                        value_loss = torch.max(surr1, surr2).mean()

                        loss = policy_loss - self.entropy_coef * entropy + self.value_loss_coef * value_loss

                        # Update batch values

                        batch_entropy += entropy.item()
                        batch_value += value.mean().item()
                        batch_policy_loss += policy_loss.item()
                        batch_value_loss += value_loss.item()
                        batch_loss += loss

                    batch_num_of_subgoals += num_of_subgoals

                # Update batch values

                batch_entropy /= batch_num_of_subgoals
                batch_value /= batch_num_of_subgoals
                batch_policy_loss /= batch_num_of_subgoals
                batch_value_loss /= batch_num_of_subgoals
                batch_loss /= batch_num_of_subgoals

                # Update actor-critic

                self.optimizer.zero_grad()
                batch_loss.backward()
                grad_norm = sum(p.grad.data.norm(2) ** 2 for p in self.acmodel.parameters() if p.grad is not None) ** 0.5
                torch.nn.utils.clip_grad_norm_(self.acmodel.parameters(), self.max_grad_norm)
                self.optimizer.step()

                # Update log values

                log_entropies.append(batch_entropy)
                log_values.append(batch_value)
                log_policy_losses.append(batch_policy_loss)
                log_value_losses.append(batch_value_loss)
                log_grad_norms.append(grad_norm.item())
                log_losses.append(batch_loss.item())

            # Log some values

            logs["entropy"] = numpy.mean(log_entropies)
            logs["value"] = numpy.mean(log_values)
            logs["policy_loss"] = numpy.mean(log_policy_losses)
            logs["value_loss"] = numpy.mean(log_value_losses)
            logs["grad_norm"] = numpy.mean(log_grad_norms)
            logs["loss"] = numpy.mean(log_losses)

        return logs

    # episode_weight_type
    #   0: each subgoal in each episode in a batch has the same weight
    #   1: each subgoal in each episode in a batch has the weight, 1/num_subgoals_in_episode
    def update_parameters_parallel_episodes(self, exps, logs):
        # Collect experiences

        num_envs = 1 # num of processes
        num_of_subgoals_per_episode = [len(actions) for actions in exps.action]

        for _ in range(self.epochs):
            # Initialize log values

            log_entropies = []
            log_values = []
            log_policy_losses = []
            log_value_losses = []
            log_grad_norms = []

            log_losses = []

            # for each epoch, we create self.num_episodes batches. One episode maps to one batch.
            episode_ids = numpy.arange(0, self.num_episodes)
            episode_ids = numpy.random.permutation(episode_ids)
            num_batches = self.num_episodes//self.num_episodes_per_batch
            for batch_group_idx in range(num_batches):
                batch_entropy = 0
                batch_value = 0
                batch_policy_loss = 0
                batch_value_loss = 0
                batch_loss = 0

                batch_start       = batch_group_idx*self.num_episodes_per_batch
                batch_after_end   = (batch_group_idx+1)*self.num_episodes_per_batch
                episode_ids_in_batch = episode_ids[batch_start:batch_after_end]
                num_of_subgoals_in_batch = [num_of_subgoals_per_episode[i] for i in episode_ids_in_batch]
                max_num_of_subgoals_in_bacth = max(num_of_subgoals_in_batch)

                num_of_subgoals_in_batch = torch.tensor(num_of_subgoals_in_batch, device=self.device).unsqueeze(dim=1)
                subgoal_mask = torch.ones(self.num_episodes_per_batch, max_num_of_subgoals_in_bacth, device=self.device)
                ascending_indices = torch.tensor([range(1,1+max_num_of_subgoals_in_bacth)]*self.num_episodes_per_batch, device=self.device)
                subgoal_mask = subgoal_mask.masked_fill(ascending_indices>num_of_subgoals_in_batch, 0)

                if self.episode_weight_type == 0:
                    subgoal_weight = subgoal_mask.clone()
                elif self.episode_weight_type == 1:
                    subgoal_weight = subgoal_mask / num_of_subgoals_in_batch
                else:
                    raise(Exception(f"Unsupported episode_weight_type, {self.episode_weight_type}"))


                value_subgoals = torch.zeros(self.num_episodes_per_batch, max_num_of_subgoals_in_bacth, device=self.device)
                entropy_subgoals = torch.zeros(self.num_episodes_per_batch, max_num_of_subgoals_in_bacth, device=self.device)
                log_prob_subgoals = torch.zeros(self.num_episodes_per_batch, max_num_of_subgoals_in_bacth, device=self.device)

                ep_log_prob = torch.zeros(self.num_episodes_per_batch, max_num_of_subgoals_in_bacth, device=self.device)
                ep_advantage = torch.zeros(self.num_episodes_per_batch, max_num_of_subgoals_in_bacth, device=self.device)
                ep_value = torch.zeros(self.num_episodes_per_batch, max_num_of_subgoals_in_bacth, device=self.device)
                ep_returnn = torch.zeros(self.num_episodes_per_batch, max_num_of_subgoals_in_bacth, device=self.device)

                for idx, ep_id in enumerate(episode_ids[batch_start:batch_after_end]):

                    num_of_subgoals = num_of_subgoals_per_episode[ep_id]
                    # Create an episode of experience
                    ep = exps[ep_id]

                    model_results = self.acmodel(ep.obs, record_subgoal_time_step=True)
                    dist = model_results['dist']
                    raw_value = model_results['value']
                    # subgoal_indice_per_sample: list of lists
                    # batch_size (number of processes) = length of input_ids_len. (it is 1 for now)
                    # number of subgoals in each episode i = input_ids_len[0][i]. batch_size=1
                    subgoal_indice_per_sample = model_results['subgoal_indice_per_sample']
                    # input_ids_len: 1-D tensor that stores indices of the recent subgoals in each process
                    input_ids_len = model_results['input_ids_len']

                    raw_entropy = dist.entropy()
                    raw_log_prob = dist.log_prob(torch.cat(ep.action, dim=0).unsqueeze(dim=1))

                    # currently support one process/one environment
                    value_subgoals[idx, :num_of_subgoals] = raw_value[range(num_envs), subgoal_indice_per_sample[0]]
                    entropy_subgoals[idx, :num_of_subgoals] = raw_entropy[range(num_envs), subgoal_indice_per_sample[0]]
                    log_prob_subgoals[idx, :num_of_subgoals] = raw_log_prob[range(num_of_subgoals), subgoal_indice_per_sample[0]]
                    
                    ep_log_prob[idx, :num_of_subgoals] = torch.cat(ep.log_prob, dim=0)
                    ep_advantage[idx, :num_of_subgoals] = torch.cat(ep.advantage, dim=0)
                    ep_value[idx, :num_of_subgoals] = torch.cat(ep.value, dim=0)
                    ep_returnn[idx, :num_of_subgoals] = torch.cat(ep.returnn, dim=0)

                ratio = torch.exp(log_prob_subgoals - ep_log_prob)
                surr1 = ratio * ep_advantage
                surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * ep_advantage
                policy_loss = -torch.min(surr1, surr2)

                value_clipped = ep_value + torch.clamp(value_subgoals - ep_value, -self.clip_eps, self.clip_eps)
                surr1 = (value_subgoals - ep_returnn).pow(2)
                surr2 = (value_clipped - ep_returnn).pow(2)
                value_loss = torch.max(surr1, surr2)

                loss = policy_loss - self.entropy_coef * entropy_subgoals + self.value_loss_coef * value_loss
                loss = loss*subgoal_weight*subgoal_mask
                loss = (loss.sum(dim=0)/subgoal_mask.sum(dim=0)).mean()

                # Update batch values

                batch_entropy += (entropy_subgoals.sum(dim=0)/subgoal_mask.sum(dim=0)).mean().item()
                batch_value += (value_subgoals.sum(dim=0)/subgoal_mask.sum(dim=0)).mean().item()
                batch_policy_loss += (policy_loss.sum(dim=0)/subgoal_mask.sum(dim=0)).mean().item()
                batch_value_loss += (value_loss.sum(dim=0)/subgoal_mask.sum(dim=0)).mean().item()
                batch_loss += loss


                # Update actor-critic

                self.optimizer.zero_grad()
                batch_loss.backward()
                grad_norm = sum(p.grad.data.norm(2) ** 2 for p in self.acmodel.parameters() if p.grad is not None) ** 0.5
                torch.nn.utils.clip_grad_norm_(self.acmodel.parameters(), self.max_grad_norm)
                self.optimizer.step()

                # Update log values

                log_entropies.append(batch_entropy)
                log_values.append(batch_value)
                log_policy_losses.append(batch_policy_loss)
                log_value_losses.append(batch_value_loss)
                log_grad_norms.append(grad_norm.item())
                log_losses.append(batch_loss.item())

            # Log some values

            logs["entropy"] = numpy.mean(log_entropies)
            logs["value"] = numpy.mean(log_values)
            logs["policy_loss"] = numpy.mean(log_policy_losses)
            logs["value_loss"] = numpy.mean(log_value_losses)
            logs["grad_norm"] = numpy.mean(log_grad_norms)
            logs["loss"] = numpy.mean(log_losses)

        return logs

class PPOAlgoFlamingoHRLIL(BaseAlgoFlamingoHRLIL):
    """The class for the Proximal Policy Optimization algorithm
    ([Schulman et al., 2015](https://arxiv.org/abs/1707.06347))."""

    def __init__(self, envs, acmodel, discount=0.99, lr=7e-4, beta1=0.9, beta2=0.999,
                 gae_lambda=0.95,
                 entropy_coef=0.01, value_loss_coef=0.5, max_grad_norm=0.5,
                 adam_eps=1e-5, clip_eps=0.2, epochs=4, preprocess_obss=None,
                 reshape_reward=None, agent=None, num_episodes=None, expert_model=None):
        num_episodes = num_episodes or 10

        super().__init__(envs, acmodel, discount, lr, gae_lambda, entropy_coef,
                         value_loss_coef, max_grad_norm, preprocess_obss, reshape_reward,
                         agent=agent, num_episodes=num_episodes, expert_model=expert_model)

        self.clip_eps = clip_eps
        self.epochs = epochs

        self.optimizer = torch.optim.Adam(self.acmodel.parameters(), lr, (beta1, beta2), eps=adam_eps)

    def update_parameters(self):
        # Collect experiences

        exps, logs = self.collect_experiences()
        '''
        exps is a DictList with the following keys: 'obs', 'expert_actions'. Each attribute, e.g.,
        `exps.expert_actions` is a list with a length of self.num_episode and each of its element is a list
        represents the expert's action for the observation at a time step. Here, the time step corresponds to the
        high-level policy in the HRL. That is, it is the time point when the corresponding subgoal is done.

        '''
        num_envs = 1 # num of processes

        for _ in range(self.epochs):
            # Initialize log values

            log_entropies = []
            log_values = []
            log_policy_losses = []
            log_value_losses = []

            log_grad_norms = []
            log_losses = []


            # for each epoch, we create self.num_episodes batches. One episode maps to one batch.
            episode_ids = numpy.arange(0, self.num_episodes)
            episode_ids = numpy.random.permutation(episode_ids)
            for ep_idx in episode_ids:

                batch_entropy = 0
                batch_value = 0
                batch_policy_loss = 0
                batch_value_loss = 0

                batch_loss = 0

                # Create an episode of experience
                ep = exps[ep_idx]

                model_results = self.acmodel(ep.obs, record_subgoal_time_step=True)
                # subgoal_indice_per_sample: list of lists
                # batch_size (number of processes) = length of input_ids_len. (it is 1 for now)
                # number of subgoals in each episode i = input_ids_len[0][i]. batch_size=1
                subgoal_indice_per_sample = model_results['subgoal_indice_per_sample']
                # input_ids_len: 1-D tensor that stores indices of the recent subgoals in each process
                input_ids_len = model_results['input_ids_len']

                raw_entropy = model_results['dist'].entropy()
                raw_logits = model_results['logits']
                raw_values = model_results['value']

                # currently support one process/one environment
                agent_entropy = raw_entropy[range(num_envs), subgoal_indice_per_sample[0]]
                agent_values = raw_values[range(num_envs), subgoal_indice_per_sample[0]]
                agent_logits = raw_logits[range(num_envs), subgoal_indice_per_sample[0], :] 
                expert_actions = torch.cat(ep.expert_actions, dim=0)
                expert_values = torch.cat(ep.expert_values, dim=0)

                loss_fn = torch.nn.CrossEntropyLoss(reduction='mean')
                policy_loss = loss_fn(agent_logits, expert_actions)
                value_loss = (expert_values - agent_values).pow(2).mean()
                entropy = agent_entropy.mean()
                value = agent_values.mean()

                batch_loss = policy_loss + self.value_loss_coef*value_loss
                batch_policy_loss = policy_loss
                batch_value_loss = value_loss
                batch_entropy = entropy
                batch_value = value

                # Update actor

                self.optimizer.zero_grad()
                batch_loss.backward()
                grad_norm = sum(p.grad.data.norm(2) ** 2 for p in self.acmodel.parameters() if p.grad is not None) ** 0.5
                torch.nn.utils.clip_grad_norm_(self.acmodel.parameters(), self.max_grad_norm)
                self.optimizer.step()

                # Update log values

                log_entropies.append(batch_entropy.item())
                log_values.append(batch_value.item())
                log_policy_losses.append(batch_policy_loss.item())
                log_value_losses.append(batch_value_loss.item())

                log_grad_norms.append(grad_norm.item())
                log_losses.append(batch_loss.item())

        # Log some values

        logs["entropy"] = numpy.mean(log_entropies)
        logs["value"] = numpy.mean(log_values)
        logs["policy_loss"] = numpy.mean(log_policy_losses)
        logs["value_loss"] = numpy.mean(log_value_losses)
        logs["grad_norm"] = numpy.mean(log_grad_norms)
        logs["loss"] = numpy.mean(log_losses)

        return logs
