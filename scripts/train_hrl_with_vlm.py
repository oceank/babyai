#!/usr/bin/env python3

"""
Script to train the agent through reinforcment learning.
"""

import os
import logging
import csv
import json
import gym
import time
import datetime
import torch
import numpy as np
import subprocess

import babyai
import babyai.utils as utils
import babyai.rl
from babyai.arguments import ArgumentParser
from babyai.model import ACModel, FlamingoACModel
from babyai.evaluate import batch_evaluate, batch_evaluate_hrl_agent
from babyai.utils.agent import ModelAgent, SkillModelAgent
from gym_minigrid.wrappers import RGBImgPartialObsWrapper

from transformers import GPT2LMHeadModel, GPT2Tokenizer, top_k_top_p_filtering, get_linear_schedule_with_warmup
from vit_pytorch.vit import ViT
from vit_pytorch.extractor import Extractor

from FlamingoGPT2.model import FlamingoGPT2
from FlamingoGPT2.utils import * # train, visualize_training_stats, load_processed_raw_data, prepare_dataload
from einops import rearrange

from babyai.utils.model import create_random_hrl_vlm_model, load_model
from babyai.levels.verifier import LowlevelInstrSet

# Parse arguments
parser = ArgumentParser()
parser.add_argument("--algo", default='ppo',
                    help="algorithm to use (default: ppo)")
parser.add_argument("--discount", type=float, default=0.99,
                    help="discount factor (default: 0.99)")
parser.add_argument("--reward-scale", type=float, default=20.,
                    help="Reward scale multiplier")
parser.add_argument("--gae-lambda", type=float, default=0.99,
                    help="lambda coefficient in GAE formula (default: 0.99, 1 means no gae)")
parser.add_argument("--value-loss-coef", type=float, default=0.5,
                    help="value loss term coefficient (default: 0.5)")
parser.add_argument("--max-grad-norm", type=float, default=0.5,
                    help="maximum norm of gradient (default: 0.5)")
parser.add_argument("--clip-eps", type=float, default=0.2,
                    help="clipping epsilon for PPO (default: 0.2)")
parser.add_argument("--ppo-epochs", type=int, default=4,
                    help="number of epochs for PPO (default: 4)")
parser.add_argument("--use-pixel", action="store_true", default=False,
                    help="the input visual observation to the acmodel is in RGB pixel.")
parser.add_argument("--save-interval", type=int, default=50,
                    help="number of updates between two saves (default: 50, 0 means no saving)")
# HRL - VLM
parser.add_argument("--use-vlm", action="store_true", default=False,
                    help="use a visual-language model (VLM) to assist the RL agent")
parser.add_argument("--max-history-window-vlm", type=int, default=16,
                    help="maximum number of observations that can be hosted in the history for VLM (default: 16)")
parser.add_argument("--max-lang-model-input-len", type=int, default=1024,
                    help="maximum number of tokens in one sequence that the VLM's language model can handel (default: 1024)")
parser.add_argument("--max-desc-len", type=int, default=20,
                    help="maxmium number of tokens in a newly generated sentence (default: 20)")
parser.add_argument("--abstract-history", action="store_true", default=False,
                    help="Allows you to switch between the full history and the abstraction of the full history")
parser.add_argument("--only-attend-immediate-media", action="store_true", default=False,
                    help="The VLM has a text token only attend to its immediately previous media. The true value of this argumetn will make the non-immediate media collected in the full history mode useless.")
'''
parser.add_argument("--top-k", type=int, default=50,
                    help="The number of tokens with the top predicted probabilities (default: 50)")
parser.add_argument("--top-p", type=float, default=0.95,
                    help="The group of tokens where the summation of their predicted probabilities is >= <top-p> (default: 0.95)")
parser.add_argument("--sample-next-token", action="store_true", default=False,
                    help="Get the next token by sampling the predicted probability distribution over the vocabulary (default: True)")
'''
# Skill Library
parser.add_argument("--skill-names-file", type=str, default="skill_model_names.txt",
                    help="File containing the names of the skills to be used for the mission")
parser.add_argument("--skill-instr-arch", default="gru",
                    help="arch to encode instructions, possible values: gru, bigru, conv, bow (default: gru)")
parser.add_argument("--skill-arch", default='bow_endpool_res',
                    help="image embedding architecture")
parser.add_argument("--skill-budget-steps", type=int, default=24,
                    help="the maximum number of steps allowed for each skill (default: 24)")
# Training Modes
parser.add_argument("--num-episodes", type=int, default=4,
                    help="number of episodes on procesess will run to collect experience before model update. used in HRL & VLM.")
parser.add_argument("--num-episodes-per-batch", type=int, default=4,
                    help="number of episodes in each batch that is used for one update of the model. used in HRL & VLM.")

parser.add_argument("--episode-based-training", action="store_true", default=False,
                    help="use the entire episode to train the agent.")
parser.add_argument("--average-loss-by-subgoals", action="store_true", default=False,
                    help="average loss by subgoals from diferent episodes.")
parser.add_argument("--episode-weight-type", type=int, default=0,
                    help="0 indicates all subgoals has the same weight when calculating averaged loss of a batch; 1 indicates each subgoal has a weight of 1/num_subgoals_in_episode.")

parser.add_argument("--save-initial-model", action="store_true", default=False,
                    help="save the initial model to see if the model is learning anything at all")

# Use VLM to generate a sentece as the subgoal
parser.add_argument("--generate-subgoal-desc", action="store_true", default=False,
                    help="Use the VLM to generate a sentence as the subgoal.")

args = parser.parse_args()
device = "cuda" if torch.cuda.is_available() else "cpu"
utils.seed(args.seed)

# Generate environments
print(f"===>  Generate {args.procs} instances of {args.env} environment")
envs = []
use_pixel = args.use_pixel
for i in range(args.procs):
    env = gym.make(args.env)
    if use_pixel:
        env = RGBImgPartialObsWrapper(env)
    env.seed(100 * args.seed + i)
    envs.append(env)

# Load skill library
print(f"===>    Loading skill library from {args.skill_names_file}")
skill_library = {}
skill_memory_size = None
skill_model_names_fp = os.path.join(utils.storage_dir(), "models", args.skill_names_file)
with open(skill_model_names_fp, 'r') as f:
    skill_names = f.readlines()
    skill_names = [skill_name.strip() for skill_name in skill_names]

    for skill_model_name in skill_names:
        skill = utils.load_skill(skill_model_name, args.skill_budget_steps)
        skill['model'].to(device)
        skill_library[skill['description']] = skill
    # assume all skills use the same memory size for their LSTM componenet
    skill_memory_size = skill['model'].memory_size
for skill_desc in skill_library:
    print(skill_desc)
# Initialize subgoal set
print(f"===>    Initializing the predefined subgoal set")
subgoal_set = LowlevelInstrSet()
subgoal_indices_str = [str(sidx) for sidx in range(subgoal_set.num_subgoals_info['total'])]
subgoal_set.display_all_subgoals()

# Create a random HRL-VLM model as the high-level policy if it does not exist
print(f"===>    Creating a random HRL-VLM model or load the model from {args.model}")
if args.model is None:
    lang_model_name="distilgpt2"
    num_high_level_actions = subgoal_set.num_subgoals_info['total']
    acmodel, args.model = create_random_hrl_vlm_model(
        args.env, args.seed, num_high_level_actions,
        args.skill_arch, args.instr_arch, args.max_history_window_vlm, device,
        lang_model_name=lang_model_name,
        only_attend_immediate_media=args.only_attend_immediate_media,
        abstract_history=args.abstract_history,
        max_lang_model_input_len=args.max_lang_model_input_len,)
elif isinstance(args.model, str):
    acmodel = load_model(args.model)

print(f"===>    Saving the initial model if it is requested.")
model_dir = os.path.join(utils.storage_dir(), "models", args.model)
os.makedirs(model_dir)
model_init_path = os.path.join(model_dir, "model_init.pt")
model_best_path = os.path.join(model_dir, "model_best.pt")
model_curr_path = os.path.join(model_dir, "model_curr.pt")
if args.save_initial_model:
    torch.save(acmodel, model_init_path)
torch.save(acmodel, os.path.join(model_dir, "model.pt"))

utils.configure_logging(args.model)
logger = logging.getLogger(__name__)

# Define obss preprocessor
obss_preprocessor = utils.ObssPreprocessor(args.model, envs[0].observation_space, args.pretrained_model)
# Initialize an instance of HRLAgent
# 1.    env is used to create obss_preprocessor in load_agent()
#       consider to remove it.
# 2.    argmax for training agent is set to False; set it True for evaluation
print(f"===>    Initializing the HRL agent")
train_agent = utils.load_agent(
        env=envs[0], model_name=args.model, argmax=False,
        skill_library=skill_library, skill_memory_size=skill_memory_size,
        subgoal_set=subgoal_set, use_vlm=True, abstract_history=args.abstract_history,)

# Define actor-critic algo
print(f"===>    Initializing the actor-critic algorithm")
reshape_reward = lambda _0, _1, reward, _2: args.reward_scale * reward
if args.algo == "ppo":
    # update model when a number of episodes pass
    if args.episode_based_training:
        algo = babyai.rl.PPOAlgoFlamingoHRLv1(envs, acmodel, args.discount, args.lr, args.beta1, args.beta2,
                                args.gae_lambda, args.entropy_coef, args.value_loss_coef, args.max_grad_norm,
                                args.optim_eps, args.clip_eps, args.ppo_epochs, obss_preprocessor, reshape_reward,
                                agent=train_agent, num_episodes=args.num_episodes,
                                generate_subgoal_desc=args.generate_subgoal_desc,
                                num_episodes_per_batch=args.num_episodes_per_batch,
                                average_loss_by_subgoals=args.average_loss_by_subgoals,
                                episode_weight_type=args.episode_weight_type)
    # update model when a number of subgoals finishes   
    else:
        raise NotImplementedError("The code for updating the model when a number of subgoals finishes is not implemented yet.")
else:
    raise ValueError("Incorrect algorithm name: {}".format(args.algo))


# Restore training status
status_path = os.path.join(model_dir, 'status.json')
if os.path.exists(status_path):
    with open(status_path, 'r') as src:
        status = json.load(src)
else:
    status = {'i': 0,
              'num_episodes': 0,
              'num_frames': 0,
              'num_high_level_actions': 0,}

# Define logger and Tensorboard writer and CSV writer
header = (["update", "episodes", "frames", "FPS", "duration"]
          + ["return_" + stat for stat in ['mean', 'std', 'min', 'max']]
          + ["success_rate"]
          + ["num_frames_" + stat for stat in ['mean', 'std', 'min', 'max']]
          + ["entropy", "value", "policy_loss", "value_loss", "loss", "grad_norm"]
          + ["high_level_actions"]
          + ["num_high_level_actions_" + stat for stat in ['mean', 'std', 'min', 'max']])

if args.tb:
    from tensorboardX import SummaryWriter

    writer = SummaryWriter(utils.get_log_dir(args.model))
csv_path = os.path.join(utils.get_log_dir(args.model), 'log.csv')
first_created = not os.path.exists(csv_path)
# we don't buffer data going in the csv log, cause we assume
# that one update will take much longer that one write to the log
csv_writer = csv.writer(open(csv_path, 'a', 1))
if first_created:
    csv_writer.writerow(header)

# Log code state, command, availability of CUDA and model

babyai_code = list(babyai.__path__)[0]
try:
    last_commit = subprocess.check_output(
        'cd {}; git log -n1'.format(babyai_code), shell=True).decode('utf-8')
    logger.info('LAST COMMIT INFO:')
    logger.info(last_commit)
except subprocess.CalledProcessError:
    logger.info('Could not figure out the last commit')
try:
    diff = subprocess.check_output(
        'cd {}; git diff'.format(babyai_code), shell=True).decode('utf-8')
    if diff:
        logger.info('GIT DIFF:')
        logger.info(diff)
except subprocess.CalledProcessError:
    logger.info('Could not figure out the last commit')
logger.info('COMMAND LINE ARGS:')
logger.info(args)
logger.info("CUDA available: {}".format(torch.cuda.is_available()))
logger.info(acmodel)

# Train model

total_start_time = time.time()
best_success_rate = 0
best_mean_return = 0
test_env_name = args.env
# 'num_frames':
# low-level task: number of primitive steps
# high-level task: number of steps to complete some subgoals
#                  num_primitive_steps is the num_frames in low-leve task case
while status['num_frames'] < args.frames:
    # Update parameters

    update_start_time = time.time()
    logs = algo.update_parameters()
    update_end_time = time.time()

    status['num_frames'] += logs["num_frames"]
    status['num_episodes'] += logs['episodes_done']
    status['i'] += 1
    status['num_high_level_actions'] += logs['num_high_level_actions']

    # Print logs

    if status['i'] % args.log_interval == 0:
        total_ellapsed_time = int(time.time() - total_start_time)
        duration = datetime.timedelta(seconds=total_ellapsed_time)
        fps = logs["num_frames"] / (update_end_time - update_start_time)
        num_frames_per_episode = utils.synthesize(logs["num_frames_per_episode"])
        return_per_episode = utils.synthesize(logs["return_per_episode"])
        success_per_episode = utils.synthesize(
            [1 if r > 0 else 0 for r in logs["return_per_episode"]])

        num_high_level_actions_per_episode = utils.synthesize(logs["num_high_level_actions_per_episode"])

        data = [status['i'], status['num_episodes'], status['num_frames'],
                fps, total_ellapsed_time,
                *return_per_episode.values(),
                success_per_episode['mean'],
                *num_frames_per_episode.values(),
                logs["entropy"], logs["value"], logs["policy_loss"], logs["value_loss"],
                logs["loss"], logs["grad_norm"],
                status["num_high_level_actions"], *num_high_level_actions_per_episode.values()]

        format_str = ("U {} | E {} | F {:06} | FPS {:04.0f} | D {} | R:xsmM {: .2f} {: .2f} {: .2f} {: .2f} | "
                      "S {:.2f} | F:xsmM {:.1f} {:.1f} {} {} | H {:.3f} | V {:.3f} | "
                      "pL {: .3f} | vL {:.3f} | L {:.3f} | gN {:.3f} | "
                      "hla {:06} | hla:xsmM {:.1f} {:.1f} {} {} | ")

        logger.info(format_str.format(*data))
        if args.tb:
            assert len(header) == len(data)
            for key, value in zip(header, data):
                writer.add_scalar(key, float(value), status['num_frames'])

        csv_writer.writerow(data)

    if args.save_interval > 0 and status['i'] % args.save_interval == 0:
        torch.save(acmodel, os.path.join(model_dir, "model.pt"))

    '''
    if args.save_interval > 0 and status['i'] % args.save_interval == 0:
        with open(status_path, 'w') as dst:
            json.dump(status, dst)
            torch.save(acmodel, model_curr_path)

        # Testing the model before saving
        agent = utils.load_agent(
            env=envs[0], model_name=args.model, argmax=False,
            skill_library=skill_library, skill_memory_size=skill_memory_size,
            subgoal_set=subgoal_set, use_vlm=True,)
        # args.model may have historys of previous training. need to clear it before evaluation

        agent.model.eval()

        logs = batch_evaluate_hrl_agent(
            agent,
            test_env_name,
            args.val_seed,
            args.val_episodes,
            pixel=use_pixel,
            concurrent_episodes=args.val_concurrent_episodes,
            use_subgoal=args.use_subgoal)

        agent.model.train()

        mean_return = np.mean(logs["return_per_episode"])
        success_rate = np.mean([1 if r > 0 else 0 for r in logs['return_per_episode']])
        save_model = False
        if success_rate > best_success_rate:
            best_success_rate = success_rate
            save_model = True
        elif (success_rate == best_success_rate) and (mean_return > best_mean_return):
            best_mean_return = mean_return
            save_model = True
        if save_model:
            torch.save(acmodel, model_best_path)
            logger.info("Return {: .2f}; best model is saved".format(mean_return))
        else:
            logger.info("Return {: .2f}; not the best model; not saved".format(mean_return))
    '''

print(f"Total time elapsed: {time.time() - total_start_time}")
