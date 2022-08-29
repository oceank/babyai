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

    def __init__(self, model_or_name, obss_preprocessor, argmax):
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

    # subgoals: list of subgoals. Each is a dictionary that has the properties,
    #           "desc" and "instr".
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

    def __init__(self, model_or_name, obss_preprocessor, argmax, subgoals, goal, skill_library):
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

        # Currently only support training and testing with one environment instance,
        # but not multiple environment instances
        self.subgoals = subgoals
        self.goal = goal

        self.skill_library = skill_library

        for skill in self.skill_library:
            skill['model'] = utils.load_model(skill['model_name'])
            if torch.cuda.is_available():
                skill['model'].cuda()
            # load the learned vocab of the skill and use it to tokenize the subgoal
            skill["obss_preprocessor"] = utils.ObssPreprocessor(skill['model_name'])
            skill["budget_steps"] = 25 # each skill will roll out 25 steps at most

        # assume all skills use the same memory size for their LSTM componenet
        self.skill_memory_size = self.skill_library[0]['model'].memory_size

        self.current_subgoal_idx = None
        self.current_skill = None
        self.current_subgoal_instr = None
        self.current_subgoal_desc  = None
        self.current_subgoal_obss_preprocessor = None
        self.current_subgoal_budget_steps = None

    # called by the reset() of an environment
    def update_subgoal_desc_and_instr(self, idx, desc, instr):
        self.subgoals[idx]['desc'] = desc
        self.subgoals[idx]['instr'] = instr

    def select_new_subgoal(self, new_subgoal_idx):
        self.current_subgoal_idx = new_subgoal_idx
        self.setup_serving_subgoal(new_subgoal_idx)

    def setup_serving_subgoal(self, subgoal_idx=0):
        self.current_subgoal_idx = subgoal_idx

        self.current_skill = self.skill_library[self.current_subgoal_idx]['model']
        self.current_subgoal_obss_preprocessor = self.skill_library[self.current_subgoal_idx]['obss_preprocessor']
        self.current_subgoal_budget_steps = self.skill_library[self.current_subgoal_idx]['budget_steps']

        self.current_subgoal_desc = self.subgoals[self.current_subgoal_idx]['desc']
        self.current_subgoal_instr = self.subgoals[self.current_subgoal_idx]['instr']

    def verify_current_subgoal(self, action):
        return self.verify_goal_completion(self.current_subgoal_idx, action)

    def verify_subgoal_completion(self, subgoal_idx, action):
        is_completed = False
        subgoal = self.subgoals[subgoal_idx]
        status = subgoal['instr'].verify(action)
        if (status == 'success'):
            print(f"===> [Subgoal Completed] {subgoal['desc']}")
            is_completed = True

        return is_completed
    
    def verify_goal_completion(self, env, action):
        reward = 0
        done = False

        if env.instrs.verify(action) == 'success': # the initial goal is completed
            done = True
            reward = env.reward()

        return reward, done

    # Currently, sequentially process the subgoal in each parallelingly running environment
    def apply_skill_batch(self, many_obs, envs, subgoal_indices):
        num_envs = len(many_obs)
        subgoals_completion = []
        memories = torch.zeros(num_envs, self.skill_memory_size, device=self.device)

        for env_idx, subgoal_idx, obs, memory in zip(range(num_envs), subgoal_indices, many_obs, memories, envs):
            env = envs[env_idx]
            steps = 0
            skill = self.skill_library[subgoal_idx]
            subgoal = env.sub_goals[subgoal_idx] #self.subgoals[subgoal_idx]
            subgoal['instr'].reset_verifier(env)
            is_completed = False
            while steps < skill['budget_steps']:
                # set the text observation (i.e., instruction) to be the description of the subgoal
                obs["mission"] = subgoal['desc']

                preprocessed_obs = skill['obss_preprocessor']([obs], device=self.device)

                with torch.no_grad():
                    model_results = skill['model'](preprocessed_obs, [memory])
                    dist = model_results['dist']
                    value = model_results['value']
                    memory = model_results['memory']

                if self.argmax:
                    action = dist.probs.argmax(1)
                else:
                    action = dist.sample()

                obs, reward, done, _ = env.step(action.cpu().numpy())

                steps += 1

                is_completed = self.verify_subgoal_completion(subgoal_idx, action)
                if is_completed:
                    break

            subgoals_completion.append(is_completed)

            # Update the next observation for the high-level policy
            # when the budget of steps are used or the subgoal is completed earlier
            many_obs[env_idx] = obs

            reward, done = self.verify_goal_completion(env, action.cpu().numpy())
            

        return many_obs, reward, done, subgoals_completion
        

    def apply_skill(self, obs, env, subgoal_idx):
        return self.apply_skill_batch([obs], [env], [subgoal_idx])

    def initialize_mission(self, env):
        # Update the agent's goal
        self.goal = {'desc':env.mission, 'instr':env.instrs}

        # Update the agent's subgoals
        print(f"List of subgoals for the mission:")
        for idx, subgoal, agent_subgoal in zip(len(self.subgoals), env.sub_goals, self.subgoals):
            print(f"*** Subgoal: {subgoal['desc']}")
            self.update_subgoal_desc_and_instr(idx, subgoal['desc'], subgoal['instr'])



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
    def __init__(self, subgoals, goal, obss_preprocessor, argmax):
        self.subgoals = subgoals

        self.goal = goal
        for subgoal in self.subgoals:
            subgoal['model'] = utils.load_model(subgoal['model_name'])
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
            self.subgoals[self.current_subgoal_idx]['model_name'])
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

    def __init__(self, demos_name, env_name, origin):
        self.demos_path = utils.get_demos_path(demos_name, env_name, origin, valid=False)
        self.demos = utils.load_demos(self.demos_path)
        self.demos = utils.demos.transform_demos(self.demos)
        self.demo_id = 0
        self.step_id = 0

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

        return {'action': self.demos[self.demo_id][self.step_id][1]}

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


def load_agent(env, model_name, demos_name=None, demos_origin=None, argmax=True, env_name=None, subgoals=None, goal=None):
    # env_name needs to be specified for demo agents
    if model_name == "SubGoalModelAgent":
        return SubGoalModelAgent(subgoals=subgoals, goal=goal, obss_preprocessor=None, argmax=argmax)
    elif model_name == 'BOT':
        return BotAgent(env)
    elif model_name is not None:
        obss_preprocessor = utils.ObssPreprocessor(model_name, env.observation_space)
        return ModelAgent(model_name, obss_preprocessor, argmax)
    elif demos_origin is not None or demos_name is not None:
        return DemoAgent(demos_name=demos_name, env_name=env_name, origin=demos_origin)
