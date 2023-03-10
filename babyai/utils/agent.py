from abc import ABC, abstractmethod
import torch
from .. import utils
from babyai.bot import Bot
from babyai.model import ACModel
from random import Random


class Agent(ABC):
    """An abstraction of the behavior of an agent. The agent is able:
    - to choose an action given an observation,
    - to analyze the feedback (i.e. reward and done state) of its action."""

    def on_reset(self):
        pass

    @abstractmethod
    def act(self, obs):
        """Propose an action based on observation.

        Returns a dict, with 'action` entry containing the proposed action,
        and optionaly other entries containing auxiliary information
        (e.g. value function).

        """
        pass

    @abstractmethod
    def analyze_feedback(self, reward, done):
        pass


class ModelAgent(Agent):
    """A model-based agent. This agent behaves using a model."""

    def __init__(self, model_or_name, obss_preprocessor, argmax, model_version="current"):
        if obss_preprocessor is None:
            assert isinstance(model_or_name, str)
            obss_preprocessor = utils.ObssPreprocessor(model_or_name, model_version=model_version)
        self.obss_preprocessor = obss_preprocessor
        if isinstance(model_or_name, str):
            self.model = utils.load_model(model_or_name, model_version=model_version)
            if torch.cuda.is_available():
                self.model.cuda()
        else:
            self.model = model_or_name
        self.device = next(self.model.parameters()).device
        self.argmax = argmax
        self.memory = None

    def act_batch(self, many_obs):
        if self.memory is None:
            self.memory = torch.zeros(
                len(many_obs), self.model.memory_size, device=self.device)
        elif self.memory.shape[0] != len(many_obs):
            raise ValueError("stick to one batch size for the lifetime of an agent")
        preprocessed_obs = self.obss_preprocessor(many_obs, device=self.device)

        if self.model.use_vlm: # generate a description and add to the preprocessed_obs
            # Store the goals in the header of the list, history[0]
            if len(self.model.history) == 0:
                self.model.initialize_history_with_goals(many_obs)

            self.model.update_history(preprocessed_obs)

            # generate a text description for the current visual observation
            # and update the history
            desc_text_tokens = self.model.generate_descs_and_update_histories()

            preprocessed_obs.desc = self.model.pass_descriptions_to_agent().to(self.device)

        with torch.no_grad():
            model_results = self.model(preprocessed_obs, self.memory)
            dist = model_results['dist']
            value = model_results['value']
            self.memory = model_results['memory']

        if self.argmax:
            action = dist.probs.argmax(1)
        else:
            action = dist.sample()

        return {'action': action,
                'dist': dist,
                'value': value}

    def act(self, obs):
        return self.act_batch([obs])

    def analyze_feedback(self, reward, done):
        if isinstance(done, tuple):
            for i in range(len(done)):
                if done[i]:
                    self.memory[i, :] *= 0.
        else:
            self.memory *= (1 - done)

class SkillModelAgent(ModelAgent):
    """
    An agent uses existing skills (policies) to complete a mission.
    This agent behaves using a list of models.
    Each model provides a policy to solve a corresponding subgoal
    """

    # subgoals: a list of list of subgoals. Each list of subgoals is associated to one environment.
    #           And each subgoal is a dictionary that has the properties, "desc" and "instr".
    # skill_library: list of skills. Each is a dictionary that has the properties,
    #           "model_name" and "model". The "model" is supposed to be able to
    #           solve a distribution of subgoals similar to that described by "desc".
    #           Note: skill_library[i] could solve subgoals[i]
    # Who decide the subgoals:
    #           currently created by the environmenet during reset()
    # How to decide the subgoals:
    #           currently created by collectively use the environment
    #           information and BabyAI language
    # Which skill to use at each time-step of solving the (high-level) goal

    def __init__(self, model_or_name, obss_preprocessor, argmax, subgoals, goal, skill_library, use_vlm=False, use_subgoal_desc=False):
        if obss_preprocessor is None:
            assert isinstance(model_or_name, str)
            obss_preprocessor = utils.ObssPreprocessor(model_or_name)
        self.obss_preprocessor = obss_preprocessor
        if isinstance(model_or_name, str):
            self.model = utils.load_model(model_or_name)
            if torch.cuda.is_available():
                self.model.cuda()
        else:
            self.model = model_or_name
        self.device = next(self.model.parameters()).device
        self.argmax = argmax
        self.memory = None

        self.use_vlm = use_vlm
        self.use_subgoal_desc = use_subgoal_desc

        # Currently only support training and testing with one environment instance,
        # but not multiple environment instances
        self.subgoals = subgoals
        self.goal = goal

        self.skill_library = skill_library

        '''
        for skill in self.skill_library:
            skill['model'] = utils.load_model(skill['model_name'])
            if torch.cuda.is_available():
                skill['model'].cuda()
            # load the learned vocab of the skill and use it to tokenize the subgoal
            skill["obss_preprocessor"] = utils.ObssPreprocessor(skill['model_name'])
            skill["budget_steps"] = 24 # each skill will roll out 24 steps at most
        '''

        # assume all skills use the same memory size for their LSTM componenet
        self.skill_memory_size = self.skill_library[0]['model'].memory_size


        self.current_subgoal_idx = None
        self.current_skill = None
        self.current_subgoal_instr = None
        self.current_subgoal_desc  = None
        self.current_subgoal_obss_preprocessor = None
        self.current_subgoal_budget_steps = None


    # called by the reset() of an environment
    def update_subgoal_desc_and_instr(self, env_idx, goal_idx, desc, instr):
        self.subgoals[env_idx][goal_idx]['desc'] = desc
        self.subgoals[env_idx][goal_idx]['desc']['instr'] = instr

    def select_new_subgoal(self, new_subgoal_idx):
        self.current_subgoal_idx = new_subgoal_idx
        self.setup_serving_subgoal(new_subgoal_idx)

    def setup_serving_subgoal(self, env_idx=0, subgoal_idx=0):
        self.current_subgoal_idx = subgoal_idx

        self.current_skill = self.skill_library[self.current_subgoal_idx]['model']
        self.current_subgoal_obss_preprocessor = self.skill_library[self.current_subgoal_idx]['obss_preprocessor']
        self.current_subgoal_budget_steps = self.skill_library[self.current_subgoal_idx]['budget_steps']

        self.current_subgoal_desc = self.subgoals[env_idx][self.current_subgoal_idx]['desc']
        self.current_subgoal_instr = self.subgoals[env_idx][self.current_subgoal_idx]['instr']

    def verify_current_subgoal(self, action):
        return self.verify_subgoal_completion(self.current_subgoal_idx, action)

    def verify_subgoal_completion(self, env_idx, subgoal_idx, action):
        is_completed = False
        subgoal = self.subgoals[env_idx][subgoal_idx]
        status = subgoal['instr'].verify(action)
        if (status == 'success'):
            #print(f"===> [Subgoal Completed] {subgoal['desc']}")
            is_completed = True

        return is_completed
    
    # 'env' here refers to one single environment
    def verify_goal_completion(self, env, action):
        reward = 0
        done = False

        if env.instrs.verify(action) == 'success': # the initial goal is completed
            done = True
            reward = env.reward()
        elif env.instrs.verify(action) == 'failure':
            done = True
            reward = 0

        return reward, done

    def initialize_mission(self, env):
        # Update the agent's goal
        self.goal = {'desc':env.mission, 'instr':env.instrs}

        # Update the agent's subgoals
        print(f"List of subgoals for the mission:")
        for idx, subgoal, agent_subgoal in zip(range(len(self.subgoals)), env.sub_goals, self.subgoals):
            print(f"*** Subgoal: {subgoal['desc']}")
            self.update_subgoal_desc_and_instr(idx, subgoal['desc'], subgoal['instr'])


    def reset_goal_and_subgoals(self, env):
        envs = env
        if not isinstance(env, list): # a single env instance
            envs = [env]

        num_envs = len(envs)
        subgoals = [None] * num_envs
        goal = [None] * num_envs
        for idx, env in enumerate(envs):
            subgoals[idx] = env.sub_goals
            goal[idx] = {'desc':env.mission, 'instr':env.instrs}
        self.subgoals = subgoals
        self.goal = goal
      
    # Functionality: determine the next primitive action for each environment
    #   For each skill, filter the correspondding observations and change the observation's
    #   instruction to be the corresponding subgoal description.
    #   Return a list of actions
    # Input:
    #   skill_indices: it is assumed to be a fixed list between two calls of get_primitive_actions()
    #                   if no subgoal is done between the two calls
    def get_primitive_actions(self, many_obs, skill_indices, skill_env_map, memories):
        actions = []
        active_env_indices = []

        for skill_indice in skill_indices:
            active_env_indices += skill_env_map[skill_indice]
            skill = self.skill_library[skill_indice]

            obss = []
            for env_idx in skill_env_map[skill_indice]:
                obs = many_obs[env_idx]
                obs["mission"] = self.subgoals[env_idx][skill_indice]['desc']
                obss.append(obs)

            preprocessed_obs = skill['obss_preprocessor'](obss, device=self.device)
            memory = memories[skill_env_map[skill_indice], :]
            with torch.no_grad():
                model_results = skill['model'](preprocessed_obs, memory)
                dist = model_results['dist']
                value = model_results['value']
                memory = model_results['memory']
            memories[skill_env_map[skill_indice], :] = memory

            if self.argmax:
                action = dist.probs.argmax(1)
            else:
                action = dist.sample()

            actions += action.cpu().tolist()

        return actions, active_env_indices


    def apply_skill(self, obs, env, subgoal_idx):
        num_envs = 1
        obs, reward, done, subgoals_consumed_steps = self.apply_skill_batch(num_envs, [obs], env, subgoal_idx)
        return obs[0], reward[0], done[0], subgoals_consumed_steps[0]
    
    # Functionality:
    #   Apply selected skill and subgoal in each parallel environment instance until whichever of the
    #   following conditions is satisfied first. We call this a completion of the high-level action.
    #       1) the subgoal is done (success or failure)
    #       2) the mission goal is done
    #       3) the budget of primitive steps for the skill execution is reached
    #   Once the agent completes the high-level action in one environment, the environment will
    #   temporarily stop taking any primitive actions until the high-level actions in other environment
    #   are completed. When high-level actions in all environment are completed, the agent will then go
    #   to decide new high-level actions for all parallely running environment. 
    # Input:
    #   many_obs: list of observations from multiple parallelly running environment instances
    #   env: env is an instance of class ParallelEnv
    #   subgoal_indices: 1-D numpy array.
    #   skill_selections:  list of skill IDs (i.e., indices). Each skill is purposed to solve
    #                      a subtask in the corresponding environment instance. It will be useful
    #                      when using subgoal description to match skill description
    # Output:
    #   obs: list of observations
    #   reward: list of rewards
    #   done: list of 'done' status
    #   subgoals_consumed_steps: list of each subgoal's consumed steps 
    # Questions:
    #   1. does the agent need to track the reset mission goal and subgoals in each environment
    #   2. how to temporarily halt an environment when the assigned high-level action is completed
    def apply_skill_batch(self, num_envs, many_obs, env, subgoal_indices):
        is_setup = env.setup_subgoal(subgoal_indices)

        active_env_indices = range(num_envs)

        # skill indice == subgoal indice. This will be changed once skill and subgoal is differentiated
        skill_indice_per_env = subgoal_indices
        env0_skill_indice = subgoal_indices[0]
        skill_env_map = {}
        for subgoal_indice, env_idx in zip(subgoal_indices, active_env_indices):
            if subgoal_indice in skill_env_map:
                skill_env_map[subgoal_indice].append(env_idx)
            else:
                skill_env_map[subgoal_indice] = [env_idx]
        
        # ensure the env.envs[0] is at the first location of the list, active_env_indices,
        # if it is still active. This is to facility the processing in step() in penv.py.
        #active_env_indices = skill_env_map[env0_skill_indice].copy()
        skill_indices = [env0_skill_indice]
        for skill_indice in skill_env_map:
            if skill_indice != env0_skill_indice:
                #active_env_indices += skill_env_map[skill_indice]
                skill_indices.append(skill_indice)
        
        subgoals_consumed_steps = [0]*num_envs
        memories = torch.zeros(num_envs, self.skill_memory_size, device=self.device)
        rewards = [0]*num_envs
        dones = [False]*num_envs

        elapsed_steps = 0
        num_active_envs = num_envs
        while num_active_envs > 0: # not all high-level actions completed
            # determine the next primitive action for each environment
            actions, active_env_indices =  self.get_primitive_actions(many_obs, skill_indices, skill_env_map, memories)

            # call step() to perform one primitive step using a skill in one env
            # env is an instance of class ParallelEnv
            # The returned 'info' indicates if the subgoal is done
            obs, reward, done, info = env.step(actions, active_env_indices)
            elapsed_steps += 1

            # analyze if completion status of high-level actions
            # check if the mission in any active envirionment is done
            #many_obs = list(many_obs)
            for idx, active_env_indice in zip(range(num_active_envs), active_env_indices):
                many_obs[active_env_indice] = obs[idx]
                halt_the_env = False
                skill_ID = skill_indice_per_env[active_env_indice]
                if done[idx]:
                    dones[active_env_indice] = True
                    rewards[active_env_indice] = reward[idx]
                    self.subgoals[active_env_indice] = env.envs[active_env_indice].sub_goals
                    halt_the_env = True
                
                if (not halt_the_env) and (info[idx] or elapsed_steps == self.skill_library[skill_ID]['budget_steps']):
                    halt_the_env = True
                
                if halt_the_env:
                    subgoals_consumed_steps[active_env_indice] = elapsed_steps
                    skill_env_map[skill_ID].remove(active_env_indice)
                    if len(skill_env_map[skill_ID]) == 0:
                        del skill_env_map[skill_ID]
                        skill_indices.remove(skill_ID)
                    num_active_envs -= 1
            #many_obs = tuple(many_obs)

        return many_obs, rewards, dones, subgoals_consumed_steps
        
    def analyze_feedback(self, reward, done):
        if self.use_vlm:
            for i in range(len(done)):
                if done[i]:
                    self.memory[i] = []
        else:
            super().analyze_feedback(reward, done)

    def act_batch(self, many_obs):

        if self.use_vlm:
            if self.memory is None:
                self.memory = [[many_obs[i]] for i in range(len(many_obs))]
            elif len(self.memory) != len(many_obs):
                raise ValueError("stick to one batch size for the lifetime of an agent")
            else:
                for i in range(len(many_obs)):
                    self.memory[i].append(many_obs[i])

            
            for i in range(len(self.memory)):
                obss_to_now = self.memory[i]
                with torch.no_grad():
                    # pass all observed up to now in the episode i
                    model_results = self.model(obss_to_now, record_subgoal_time_step=True, use_subgoal_desc=self.use_subgoal_desc)
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



                if self.argmax:
                    raw_action = dist.probs.argmax(dim=-1)
                else:
                    # (b=1, max_lang_model_input_len)
                    raw_action = dist.sample()
                
                if i == 0:
                    # (b=1, ): the indice of the last token of the recent subgoal description
                    action = raw_action[range(1), input_ids_len]
                    value = raw_value[range(1), input_ids_len]
                else:
                    action = torch.cat([action, raw_action[range(1), input_ids_len]], dim=-1)
                    value  = torch.cat([value, raw_value[range(1),   input_ids_len]], dim=-1)
    
            result = {'action': action, 'value': value}

        else:
            result = super().act_batch(many_obs)

        return result

class SubGoalModelAgent(ModelAgent):
    """A subgoal-models-based agent. This agent behaves using a list of models. Each model provides a policy to solve the corresponding subgoal"""

    # subgoals: list of subgoals. Each is a dictionary that has the properties,
    #           "desc", "instr", "model_name", "model". The "model" is supposed
    #           to be able to solve a distribution of subgoals similar to that 
    #           described by "desc".
    # Who decide the subgoals:
    #           currently created by the environmenet during reset()
    # How to decide the subgoals:
    #           currently created by collectively use the environment
    #           information and BabyAI language
    def __init__(self, subgoals, goal, obss_preprocessor, argmax, model_version='current'):
        self.subgoals = subgoals

        self.goal = goal
        for subgoal in self.subgoals:
            subgoal['model'] = utils.load_model(subgoal['model_name'], model_version='best')
            if torch.cuda.is_available():
                subgoal['model'].cuda()

        self.current_subgoal_idx = None
        self.model = None
        self.current_subgoal_instr = None
        self.current_subgoal_desc  = None
        self.obss_preprocessor     = None

        #self.device = next(self.model.parameters()).device
        # Use the first gpu if it exists
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.argmax = argmax
        self.memory = None

    
    # called by the reset() of an environment
    def update_subgoal_desc_and_instr(self, idx, desc, instr):
        self.subgoals[idx]['desc'] = desc
        self.subgoals[idx]['instr'] = instr

    def select_new_subgoal(self, new_subgoal_idx):
        self.current_subgoal_idx = new_subgoal_idx
        self.setup_serving_subgoal(new_subgoal_idx)

    def setup_serving_subgoal(self, subgoal_idx=0):
        self.current_subgoal_idx = subgoal_idx
        self.model = self.subgoals[self.current_subgoal_idx]['model']
        self.obss_preprocessor = utils.ObssPreprocessor(
            self.subgoals[self.current_subgoal_idx]['model_name'], model_version='best')
        self.current_subgoal_desc = self.subgoals[self.current_subgoal_idx]['desc']
        self.current_subgoal_instr = self.subgoals[self.current_subgoal_idx]['instr']

    def verify_current_subgoal(self, action):
        is_completed = False

        status = self.current_subgoal_instr.verify(action)
        if (status == 'success'):
            print(f"===> [Subgoal Completed] {self.current_subgoal_desc}")
            is_completed = True

        return is_completed
    
    def verify_goal_completion(self, env, action):
        reward = 0
        done = False

        if env.instrs.verify(action) == 'success': # the initial goal is completed
            done = True
            reward = env.reward()

        return reward, done
           
    def reinitialize_mission(self, env):
        # Update the agent's goal
        self.goal = {'desc':env.mission, 'instr':env.instrs}

        # Update the agent's subgoals
        print(f"List of subgoals for the mission:")
        for subgoal, agent_subgoal in zip(env.sub_goals, self.subgoals):
            print(f"*** Subgoal: {subgoal['desc']}")
            agent_subgoal['instr'] = subgoal['instr']
            agent_subgoal['desc']  = subgoal['desc']


class RandomAgent:
    """A newly initialized model-based agent."""

    def __init__(self, seed=0, number_of_actions=7):
        self.rng = Random(seed)
        self.number_of_actions = number_of_actions

    def act(self, obs):
        action = self.rng.randint(0, self.number_of_actions - 1)
        # To be consistent with how a ModelAgent's output of `act`:
        return {'action': torch.tensor(action),
                'dist': None,
                'value': None}


class DemoAgent(Agent):
    """A demonstration-based agent. This agent behaves using demonstrations."""

    def __init__(self, demos_name, env_name, origin, check_subgoal_completion=False):
        self.demos_path = utils.get_demos_path(demos_name, env_name, origin, valid=False)
        self.demos = utils.load_demos(self.demos_path)
        self.demos = utils.demos.transform_demos(self.demos, check_subgoal_completion)
        self.demo_id = 0
        self.step_id = 0
        self.check_subgoal_completion = check_subgoal_completion

    @staticmethod
    def check_obss_equality(obs1, obs2):
        if not(obs1.keys() == obs2.keys()):
            return False
        for key in obs1.keys():
            if type(obs1[key]) in (str, int):
                if not(obs1[key] == obs2[key]):
                    return False
            else:
                if not (obs1[key] == obs2[key]).all():
                    return False
        return True

    def act(self, obs):
        if self.demo_id >= len(self.demos):
            raise ValueError("No demonstration remaining")
        expected_obs = self.demos[self.demo_id][self.step_id][0]
        assert DemoAgent.check_obss_equality(obs, expected_obs), "The observations do not match"

        result = {'action': self.demos[self.demo_id][self.step_id][1]}
        if self.check_subgoal_completion:
            result['completed_subgoals'] = self.demos[self.demo_id][self.step_id][3]
        return result

    def analyze_feedback(self, reward, done):
        self.step_id += 1

        if done:
            self.demo_id += 1
            self.step_id = 0


class BotAgent:
    def __init__(self, env):
        """An agent based on a GOFAI bot."""
        self.env = env
        self.on_reset()

    def on_reset(self):
        self.bot = Bot(self.env)

    def act(self, obs=None, update_internal_state=True, *args, **kwargs):
        action = self.bot.replan()
        return {'action': action}

    def analyze_feedback(self, reward, done):
        pass

class HRLAgentHistory():
    '''
    This class is used to store the history of a HRL agent.
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
    '''
    def __init__(
        self,
        goal=None,
        token_seqs=None,
        vis_obss=None,
        highlevel_actions=None,
        highlevel_timesteps=None,
        hla_hid_indices=None,
        subgoals_status=None,
        lowlevel_actions=None,
        rewards=None,
        dones=None,):
        self.goal = goal
        self.token_seqs = token_seqs
        self.vis_obss = vis_obss
        self.highlevel_actions = highlevel_actions
        self.highlevel_time_steps = highlevel_timesteps
        self.hla_hid_indices = hla_hid_indices
        self.subgoals_status = subgoals_status
        self.lowlevel_actions = lowlevel_actions
        self.rewards = rewards
        self.dones = dones
        self.token_seq_lens = None
        if self.token_seqs is not None:
            self.token_seq_lens = self.token_seqs['attention_mask'].sum(dim=1)

class HRLAgent(ModelAgent):
    """
    Major modules of the HRL agent:
    1   High-Level Policy Module:
        *   When there is an incoming request of a new subgoal,  the agent proposes the next
            subgoal based on the summary of the agent's observational history up to now.
        *   The new subgoal is then used to select a skill from the skill library.
    2   Low-Level Policy Module:
        *   According to the new subgoal and the agent's observational history from the
            start of the new subgoal to the current time step, the selected skill outputs
            a primitive action at each time step until the new subgoal is done.
    3   New-Subgoal Request Module:
        *   Request a new sugboal when any of the following conditions is met:
            1.  mission starts
            2.  the current subgoal is done
    4   History Summarization/Abstraction Module (TBD):
        *   Summarize the agent's observational history into a compact representation.
    5   Kowledge Of World Module (TBD):
        *   Provide the agent with the learned knowledge of the world.

    Data Members:
        skill_library: a dictionary where each elem is a skill. Each skill itself is a dictionary that has the properties:
            description
            model_name
            model: policy model
            obss_preprocessor
            budget_steps
        subgoal_set: an instance of Class LowlevelInstrSet that contains a set of instructions for subgoals
        history: an instance of Class HRLAgentHistory
        current_skill:
        current_skill_memory:
        current_subgoal:
        current_subgoal_status:
        current_subgoal_start_time:
        current_time_step:
    """
    def __init__(
        self, model_or_name, obss_preprocessor, argmax,
        skill_library, skill_memory_size, subgoal_set,
        use_vlm=True, abstract_history=False, only_attend_immediate_media=True,
        model_version='current'):

        self.skill_library = skill_library
        self.subgoal_set = subgoal_set
        self.model_version=model_version

        # for the high-level policy module
        if obss_preprocessor is None:
            assert isinstance(model_or_name, str)
            obss_preprocessor = utils.ObssPreprocessor(model_or_name, model_version=model_version)
        self.obss_preprocessor = obss_preprocessor
        # 1: Load the 'recent' version of model from model name
        # 2: Load the 'best' versin of model from model
        # ToDo: add a model_version as a parameter to the constructor
        if isinstance(model_or_name, str):
            self.model = utils.load_model(model_or_name, model_version=model_version)
            if torch.cuda.is_available():
                self.model.cuda()
        else:
            self.model = model_or_name

        # Currently only support training and testing with one environment instance
        self.goal = None
        self.current_time_step = None

        # For the high-level policy module
        self.abstract_history = abstract_history
        self.only_attend_immediate_media = only_attend_immediate_media
        self.num_highlevel_actions = subgoal_set.num_subgoals_info['total']
        self.history = None

        # For the low-level policy module
        self.skill_memory_size = skill_memory_size

        # Reset/Initialize data members related to current skill and current subgoal to 'None'
        self.reset_subgoal_and_skill()
    
        # Other data members
        self.debug = False
        self.device = next(self.model.parameters()).device
        self.argmax = argmax
        self.use_vlm = use_vlm

        if self.use_vlm:
            # the model has tokenizer data member
            # subgoal[1]: an Instr instance for the subgoal
            # '!': match the description format of a completed subgoal in the collection of demonstrations.
            self.subgoals_token_seqs = self.model.tokenizer(
                [subgoal[1].instr_desc+"!" for subgoal in self.subgoal_set.all_subgoals],
                return_tensors="pt",
                padding=True)
            self.subgoals_token_seqs.to(self.device)
            self.subgoals_token_lens = self.subgoals_token_seqs['attention_mask'].sum(1)
            subgoal_statuses_str = ["[InProgress]", "[Success]", "[Failure]"]
            self.subgoal_status_token_seqs = self.model.tokenizer(
                subgoal_statuses_str,
                return_tensors="pt",
                padding=True)
            self.subgoal_status_token_seqs.to(self.device)
            self.subgoal_status_token_lens = self.subgoal_status_token_seqs['attention_mask'].sum(1)

    def set_debug_mode(self, debug):
        self.debug = debug

    def set_goal(self, goal):
        self.goal = goal

    def get_goal(self):
        return self.goal

    def reset_history(self, goal, initial_obs):
        self.history = HRLAgentHistory()
        self.history.goal = goal
        self.history.token_seqs = self.model.tokenizer(
                self.history.goal+"|image|[start]",
                return_tensors="pt",
                padding="max_length",
                max_length=self.model.max_lang_model_input_len)
        self.history.token_seqs.to(self.device)
        self.history.token_seq_lens = self.history.token_seqs['attention_mask'].sum(dim=1)
        self.history.hla_hid_indices = [self.history.token_seq_lens[0]-1]
        self.history.vis_obss = [initial_obs]
        '''
        self.history.token_seqs['image_embeds'] = self.model.img_encoder([self.history.vis_obss])
        media_locations, label_masks, instance_weights = utils.vlm.cal_media_loc_labels_token_weights(self.history.token_seqs, device=self.device, only_media_locations=True)
        self.history.token_seqs['media_locations'] = media_locations
        '''

        self.history.highlevel_actions = []
        self.history.highlevel_time_steps = []
        self.history.subgoals_status = []
        self.history.lowlevel_actions = []
        self.history.rewards = []
        self.history.dones = []

    def reset_subgoal_and_skill(self):
        self.current_skill = None
        self.current_subgoal = None
        self.current_subgoal_idx = None
        self.current_subgoal_instr = None
        self.current_subgoal_desc  = None
        self.current_subgoal_status = None # 0: in progress, 1: success, 2: faliure
        self.current_subgoal_start_time = None
        self.current_subgoal_memory = None

    def on_reset(self, env, goal, initial_obs, propose_first_subgoal=True):
        self.current_time_step = 0
        self.goal = goal
        self.reset_subgoal_and_skill()
        self.reset_history(goal, initial_obs)
        if propose_first_subgoal:
            # propose the first subgoal
            self.propose_new_subgoal(env)

    # New-Subgoal Request Module
    #   the mission just starts
    #   the current subgoal is done
    # Assume the mission is not done yet.
    def need_new_subgoal(self):
        return self.current_time_step==0 or self.current_subgoal_status!=0

    # Accumulate envrionment information to the history immediately after the agent takes an action
    def accumulate_env_info_to_history(self, action, obs, reward, done):
        self.history.vis_obss.append(obs)
        self.history.lowlevel_actions.append(action)
        self.history.rewards.append(reward)
        self.history.dones.append(done)

    # Call this function when the current subgoal is done
    # Assume information given by the environment has been accumulated in the history after an action.
    # Call this function after verifing the current subgoal and before proposing a new subgoal.
    def update_history_with_subgoal_status(self):
        # Append the subgoal's visual obs seq ('|image...image|') to the history
        vis_obs_seq_str ="|" + "image"*(self.current_time_step-self.current_subgoal_start_time) + "|"
        subgoal_vis_token_seqs = self.model.tokenizer(
                vis_obs_seq_str,
                return_tensors="pt",
                padding=True,)
        subgoal_vis_token_seqs.to(self.device)
        subgoal_vis_token_lens = subgoal_vis_token_seqs['attention_mask'].sum(1)
        start = self.history.token_seq_lens[0]
        subgoal_vis_token_len = subgoal_vis_token_lens[0]
        end = start + subgoal_vis_token_len

        self.history.token_seqs['input_ids'][0, start:end] = subgoal_vis_token_seqs['input_ids'][0, :subgoal_vis_token_len]
        self.history.token_seqs['attention_mask'][0, start:end] = 1
        self.history.token_seq_lens[0] += subgoal_vis_token_len

        # Append the subgoal's status to the history
        start = self.history.token_seq_lens[0]
        subgoal_status_token_len = self.subgoal_status_token_lens[self.current_subgoal_status]
        end = start + subgoal_status_token_len

        self.history.token_seqs['input_ids'][0, start:end] = self.subgoal_status_token_seqs['input_ids'][self.current_subgoal_status, :subgoal_status_token_len]
        self.history.token_seqs['attention_mask'][0, start:end] = 1
        self.history.token_seq_lens[0] += subgoal_status_token_len

        # Append auxiliary information to the history. hla_hid_index corresponds the index of token, ']', in the last subgoal's status.
        self.history.hla_hid_indices.append(self.history.token_seq_lens[0]-1)
        self.history.subgoals_status.append(self.current_subgoal_status)

        '''
        # Update 'image_embeds' and 'media_locations'
        with torch.no_grad():
            self.history.token_seqs['image_embeds'] = self.model.img_encoder([self.history.vis_obss])
        media_locations, label_masks, instance_weights = utils.vlm.cal_media_loc_labels_token_weights(self.history.token_seqs, device=self.device, only_media_locations=True)
        self.history.token_seqs['media_locations'] = media_locations
        '''

    # Verify if the current subgoal is done
    #   the current subgoal is completed
    #   the budget steps are used up
    # Assume that the 'action' has been taken by the agent
    # Need to set the new hld_hid_index when the current subgoal is done after calling this funciton.
    def verify_current_subgoal(self, action):
        budget_steps_used_up = (self.current_time_step - self.current_subgoal_start_time) >= self.current_skill['budget_steps']
        status = self.current_subgoal_instr.verify(action)
        is_successful = (status == 'success')
        if is_successful:
            self.current_subgoal_status = 1
            if self.debug:
                print(f"===> [Subgoal Completed] {self.current_subgoal_desc}")
        elif budget_steps_used_up:
            self.current_subgoal_status = 2
            if self.debug:
                print(f"===> [Subgoal Failed] {self.current_subgoal_desc}")

    # Note: currently support only one environment instance
    # Based on the current self.history, decide the next highlevel action (the next subgoal and proper skill for it)
    # * First update the history with image_embeds and media_locations, then apply the model (VLM) to get the high-level action
    # * Update self.current_subgoal, self.current_skill
    # * Append the token sequence of the new subgoal to self.token_seqs
    def propose_new_subgoal(self, env, is_training=False):

        # Update 'image_embeds' and 'media_locations'
        self.history.token_seqs['image_embeds'] = self.model.img_encoder([self.history.vis_obss])
        # only_media_locations=True: only calculate media_locations
        media_locations, label_masks, instance_weights = utils.vlm.cal_media_loc_labels_token_weights(
            self.history.token_seqs, device=self.device, only_media_locations=True)
        self.history.token_seqs['media_locations'] = media_locations

        model_results = self.model(self.history)

        # Clear 'image_embeds' and 'media_locations' in GPU to save memory
        self.history.token_seqs['image_embeds'] = None
        self.history.token_seqs['media_locations'] = None

        result_idx = self.history.hla_hid_indices[-1]
        dist = model_results['dist']
        #logits = model_results['logits'][0, result_idx]

        if self.argmax:
            action = dist.probs.argmax(1)
        else:
            action = dist.sample()
        # shape of 'action': (b=1, max_lang_model_input_len)
        highlevel_action = action[0, result_idx]

        self.setup_new_subgoal_and_skill(env, highlevel_action)

        if is_training:
            raw_log_probs = dist.log_prob(action)
            log_prob = raw_log_probs[0, result_idx]
            value = model_results['value'][0, result_idx]
            return highlevel_action, log_prob, value

    def setup_new_subgoal_and_skill(self, env, highlevel_action):
        # Currently only support one env
        if isinstance(highlevel_action, torch.Tensor):
            highlevel_action = highlevel_action.item()
        # update the current subgoal and skill
        self.current_subgoal = self.subgoal_set.all_subgoals[highlevel_action]
        self.current_subgoal_idx = self.current_subgoal[0]
        self.current_subgoal_instr = self.current_subgoal[1]
        self.current_subgoal_desc = self.current_subgoal_instr.instr_desc
        self.current_subgoal_status = 0 # 0: in progress, 1: success, 2: faliure
        ##  self.current_time_step has been updated before calling this function
        self.current_subgoal_start_time = self.current_time_step
        self.current_subgoal_memory = torch.zeros(1, self.skill_memory_size, device=self.device)
        self.current_subgoal_instr.reset_verifier(env)

        skill_desc = self.current_subgoal[2]
        self.current_skill = self.skill_library[skill_desc]

        # update the history
        self.history.highlevel_time_steps.append(self.current_subgoal_start_time)
        self.history.highlevel_actions.append(highlevel_action)
        ## append the token sequence of the new subgoal to self.token_seqs
        if self.use_vlm:
            b_idx = 0
            start = self.history.token_seq_lens[b_idx]
            new_subgoal_token_len = self.subgoals_token_lens[highlevel_action]
            end = start + new_subgoal_token_len
            self.history.token_seqs['input_ids'][b_idx, start:end] = self.subgoals_token_seqs['input_ids'][highlevel_action, :new_subgoal_token_len]
            self.history.token_seqs['attention_mask'][b_idx, start:end] = 1

    '''
        Assume that neither the mission goal nor the current subgoal is done.
        Functionality: apply the current skill to suggest the next primitive action for solving the current subgoal
    '''
    def act(self, obs):
        obs['mission'] = self.current_subgoal_desc
        preprocessed_obs = self.current_skill['obss_preprocessor']([obs], device=self.device)
        with torch.no_grad():
            result = self.current_skill['model'](preprocessed_obs, self.current_subgoal_memory)
            dist = result['dist']
            value = result['value']
            self.current_subgoal_memory = result['memory']

        action = dist.probs.argmax(1)
        '''
        if self.argmax:
            action = dist.probs.argmax(1)
        else:
            action = dist.sample()
        '''
        return {'action': action,
                'dist': dist,
                'value': value}

    # TBD
    def analyze_feedback(self, reward, done):
        if isinstance(done, tuple):
            for i in range(len(done)):
                if done[i]:
                    self.current_subgoal_memory[i, :] *= 0.
        else:
            self.current_subgoal_memory *= (1 - done)

def load_agent(
        env,
        model_name, argmax=True,
        subgoals=None, goal=None,
        skill_library=None, skill_memory_size=None, subgoal_set=None, use_vlm=True, abstract_history=False, only_attend_immediate_media=True,
        demos_name=None, demos_origin=None, env_name=None, check_subgoal_completion=False,
        model_version='current'):
    # env_name needs to be specified for demo agents
    if model_name == 'BOT':
        return BotAgent(env)
    elif model_name == "SubGoalModelAgent":
        return SubGoalModelAgent(subgoals=subgoals, goal=goal, obss_preprocessor=None, argmax=argmax, model_version=model_version)
    elif model_name is not None:
        obss_preprocessor = utils.ObssPreprocessor(model_name, env.observation_space, model_version=model_version)
        if skill_library is not None:
            return HRLAgent(
                model_name, obss_preprocessor, argmax,
                skill_library, skill_memory_size, subgoal_set,
                use_vlm, abstract_history, only_attend_immediate_media,
                model_version=model_version)
        else:
            return ModelAgent(model_name, obss_preprocessor, argmax, model_version=model_version)
    elif demos_origin is not None or demos_name is not None:
        return DemoAgent(demos_name=demos_name, env_name=env_name, origin=demos_origin, check_subgoal_completion=check_subgoal_completion)
