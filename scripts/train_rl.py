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
from babyai.evaluate import batch_evaluate
from babyai.utils.agent import ModelAgent, SkillModelAgent
from gym_minigrid.wrappers import RGBImgPartialObsWrapper

from transformers import GPT2LMHeadModel, GPT2Tokenizer, top_k_top_p_filtering, get_linear_schedule_with_warmup
from vit_pytorch.vit import ViT
from vit_pytorch.extractor import Extractor

from FlamingoGPT2.model import FlamingoGPT2
from FlamingoGPT2.utils import * # train, visualize_training_stats, load_processed_raw_data, prepare_dataload
from einops import rearrange

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
parser.add_argument("--save-interval", type=int, default=50,
                    help="number of updates between two saves (default: 50, 0 means no saving)")
parser.add_argument("--use-vlm", action="store_true", default=False,
                    help="use a visual-language model (VLM) to assist the RL agent")
parser.add_argument("--max-history-window-vlm", type=int, default=16,
                    help="maximum number of observations that can be hosted in the history for VLM (default: 16)")
parser.add_argument("--max-lang-model-input-len", type=int, default=1024,
                    help="maximum number of tokens in one sequence that the VLM's language model can handel (default: 1024)")
parser.add_argument("--max-desc-len", type=int, default=20,
                    help="maxmium number of tokens in a newly generated sentence (default: 20)")
parser.add_argument("--top-k", type=int, default=50,
                    help="The number of tokens with the top predicted probabilities (default: 50)")
parser.add_argument("--top-p", type=float, default=0.95,
                    help="The group of tokens where the summation of their predicted probabilities is >= <top-p> (default: 0.95)")
parser.add_argument("--sample-next-token", action="store_true", default=False,
                    help="Get the next token by sampling the predicted probability distribution over the vocabulary (default: True)")


parser.add_argument("--use-subgoal", action="store_true", default=False,
                    help="use a SkillModelAgent and a library of skills to complete the goal")
parser.add_argument("--skill-model-name-list", type=str, default=None,
                    help="list of model name of each skill")

parser.add_argument("--num-episodes", type=int, default=4,
                    help="number of episodes on procesess will run to collect experience before model update. used in HRL & VLM.")
parser.add_argument("--num-episodes-per-batch", type=int, default=4,
                    help="number of episodes in each batch that is used for one update of the model. used in HRL & VLM.")


parser.add_argument("--use-subgoal-desc", action="store_true", default=False,
                    help="use the descripiton of previous subgoals and the mission description at each time step")
                    

parser.add_argument("--has-expert", action="store_true", default=False,
                    help="use an expert to guide the agent")
parser.add_argument("--expert-model-name", type=str, default="",
                    help="the name of the expert model")

parser.add_argument("--use-FiLM", action="store_true", default=False,
                    help="use FiLM layers to fuse the instruction embedding and the visual embedding")

parser.add_argument("--use-pixel", action="store_true", default=False,
                    help="the input visual observation to the acmodel is in RGB pixel.")

parser.add_argument("--episode-based-training", action="store_true", default=False,
                    help="use the entire episode to train the agent.")

args = parser.parse_args()

utils.seed(args.seed)

# Generate environments
envs = []
use_pixel = args.use_pixel
for i in range(args.procs):
    env = gym.make(args.env)
    if use_pixel:
        env = RGBImgPartialObsWrapper(env)
    env.seed(100 * args.seed + i)
    envs.append(env)


goal = None
subgoals = None
train_agent = None
skill_model_name_list = []
if args.use_subgoal:
    assert args.skill_model_name_list is not None
    skill_model_name_list = args.skill_model_name_list.split(',')

    '''
    num_envs = args.procs
    subgoals = [None] * num_envs
    goal = [None] * num_envs
    for idx, env in enumerate(envs):
        subgoals[idx] = env.sub_goals
        goal[idx] = {'desc':env.mission, 'instr':env.instrs}
    '''

    skill_library = []
    budget_steps = 24 # each skill will roll out 24 steps at most
    for skill_model_name in skill_model_name_list:
        skill = utils.load_skill(skill_model_name, budget_steps)
        skill_library.append(skill)


# Define model name
suffix = datetime.datetime.now().strftime("%y-%m-%d-%H-%M-%S")
instr = args.instr_arch if args.instr_arch else "noinstr"
mem = "mem" if not args.no_mem else "nomem"
vlm_info = "vlm" if args.use_vlm else "novlm"
subgoal_info = "subgoal" if args.use_subgoal else "nosubgoal"
film_info = "film" if args.use_FiLM else "nofilm" 

model_name_parts = {
    'env': args.env,
    'algo': args.algo,
    'arch': args.arch,
    'instr': instr,
    'mem': mem,
    'seed': args.seed,
    'info': '',
    'coef': '',
    'vlm' : vlm_info,
    'subgoal': subgoal_info,
    'film': film_info,
    'suffix': suffix}
default_model_name = "{env}_{algo}_{arch}_{instr}_{mem}_seed{seed}{info}{coef}_{vlm}_{subgoal}_{film}_{suffix}".format(**model_name_parts)
if args.pretrained_model:
    default_model_name = args.pretrained_model + '_pretrained_' + default_model_name
args.model = args.model.format(**model_name_parts) if args.model else default_model_name

utils.configure_logging(args.model)
logger = logging.getLogger(__name__)

# Define obss preprocessor
if 'emb' in args.arch:
    obss_preprocessor = utils.IntObssPreprocessor(args.model, envs[0].observation_space, args.pretrained_model)
else:
    obss_preprocessor = utils.ObssPreprocessor(args.model, envs[0].observation_space, args.pretrained_model)


# Parameters for VLM
vlm = None
tokenizer=None

if args.use_vlm:
    lang_model_name = "distilgpt2" # "gpt2"
    # gpt2: 12 transformer layers
    # distilgpt2: 6 transformer layers

    print(f"=== Initialize a visual-language model, FlamingoGPT2, to assist the agent ===")
    print(f"[Setup] Use VLM to help the anget to explore the grid world")
    print(f"[Setup] Create a tokenizer and {lang_model_name} language model")

    tokenizer = GPT2Tokenizer.from_pretrained(lang_model_name, return_dict=True)
    tokenizer.pad_token = tokenizer.eos_token # pad token
    tokenizer.sep_token = tokenizer.eos_token # sequence separator token

    lang_model = GPT2LMHeadModel.from_pretrained(
        lang_model_name,
        pad_token_id=tokenizer.pad_token_id,
        sep_token_id = tokenizer.sep_token_id)

    lang_model_config = lang_model.config
    dim_lang_embeds = lang_model_config.n_embd
    depth = lang_model_config.n_layer
    
    dim_img_embeds = dim_lang_embeds #128

    # first take your trained image encoder and wrap it in an adapter that returns the image embeddings
    # here we use the ViT from the vit-pytorch library
    print(f"[Setup] Create a visual encoder using ViT")
    vit = ViT(
        image_size = 56, #256,
        patch_size = 7, #32,
        num_classes = 1000,
        dim = dim_img_embeds,
        depth = 6,
        heads = 16,
        mlp_dim = 2048,
        dropout = 0.1,
        emb_dropout = 0.1
    )

    vit = Extractor(vit, return_embeddings_only = True)

    print(f"[Setup] Create a Flamingo Model")
    vlm = FlamingoGPT2(
        lang_model=lang_model,       # pretrained language model GPT2 with a language header
        dim = dim_lang_embeds,       # dimensions of the embedding
        depth = depth,               # depth of the language model
        # variables below are for Flamingo trainable modules
        heads = 8,                   # attention heads. 8, 4
        ff_mult=4,                   # 4, 2
        dim_head = 64,               # dimension per attention head. 64, 32
        img_encoder = vit,           # plugin your image encoder (this can be optional if you pass in the image embeddings separately, but probably want to train end to end given the perceiver resampler)
        media_token_id = 3,          # the token id representing the [media] or [image]
        cross_attn_every = 3,        # how often to cross attend
        perceiver_num_latents = 64,  # perceiver number of latents. 64, 32
                                     # It should be smaller than the sequence length of the image tokens
        perceiver_depth = 2,         # perceiver resampler depth
        perceiver_num_time_embeds = args.max_history_window_vlm,#16, 8
        only_attend_immediate_media=True
    )


if args.use_subgoal:
    # high-level task: unlock a door in one room
    # action 0: pickup a key that matches the door
    # action 1: open the door
    num_of_subgoals = len(skill_model_name_list)

    # when solving a high-level task
    num_of_actions = num_of_subgoals
else:
    # When solving a low-level task
    num_of_actions = envs[0].action_space.n

# Define actor-critic model
acmodel = utils.load_model(args.model, raise_not_found=False)
if acmodel is None:
    if args.pretrained_model:
        acmodel = utils.load_model(args.pretrained_model, raise_not_found=True)
    elif args.use_subgoal and args.use_vlm:
        acmodel = FlamingoACModel(
            obss_preprocessor.obs_space, num_of_actions,
            args.arch,
            # the following parameters are used for using the vlm
            vlm=vlm,
            tokenizer=tokenizer,
            max_desc_len=args.max_desc_len,
            max_lang_model_input_len=args.max_lang_model_input_len,
            max_history_window_vlm=args.max_history_window_vlm,
            top_k=args.top_k,
            top_p=args.top_p,
            sample_next_token=args.sample_next_token,
            use_pixel = args.use_pixel,
            use_FiLM=args.use_FiLM, cat_img_instr=False, only_lang_part=False
        )
    else: # args.use_FiLM (+LSTM)
        acmodel = ACModel(
            obss_preprocessor.obs_space, num_of_actions,
            args.image_dim, args.memory_dim, args.instr_dim,
            not args.no_instr, args.instr_arch, not args.no_mem, args.arch,
            # the following parameters are used for using the vlm
            use_vlm=args.use_vlm,
            vlm=vlm,
            tokenizer=tokenizer,
            max_desc_len=args.max_desc_len,
            max_lang_model_input_len=args.max_lang_model_input_len,
            max_history_window_vlm=args.max_history_window_vlm,
            top_k=args.top_k,
            top_p=args.top_p,
            sample_next_token=args.sample_next_token,
            use_FiLM = args.use_FiLM
        )

obss_preprocessor.vocab.save()
utils.save_model(acmodel, args.model)

if torch.cuda.is_available():
    acmodel.cuda()


if args.use_subgoal:
    train_agent = SkillModelAgent(
        acmodel, obss_preprocessor, argmax=True,
        subgoals=subgoals, goal=goal, skill_library=skill_library,
        use_vlm=args.use_vlm, use_subgoal_desc=args.use_subgoal_desc)

# Define actor-critic algo

reshape_reward = lambda _0, _1, reward, _2: args.reward_scale * reward
if args.algo == "ppo":
    # cases:
    # 1. args.use_subgoal + args.use_vlm
    # 2. args.use_subgoal + args.use_vlm + args.has_expert (imitation learning)
    # 3. args.use_subgoal + not arg.use_vlm + args.use_FiLM (+LSTM)
    if args.episode_based_training:
        if args.use_subgoal and args.use_vlm and args.had_expert: # imitation learning
            expert_model = utils.load_model(args.expert_model_name)
            expert_obss_preprocessor = utils.ObssPreprocessor(args.expert_model_name, envs[0].observation_space, args.pretrained_model)
            algo = babyai.rl.PPOAlgoFlamingoHRLIL(envs, acmodel, args.discount, args.lr, args.beta1, args.beta2,
                                    args.gae_lambda, args.entropy_coef, args.value_loss_coef, args.max_grad_norm,
                                    args.optim_eps, args.clip_eps, args.ppo_epochs, expert_obss_preprocessor,
                                    reshape_reward, agent=train_agent, num_episodes=args.num_episodes,
                                    expert_model = expert_model)
 
        #use_FiLM=False, cat_img_instr=False, only_lang_part=False
        # case 1 and case 3
        else:
            algo = babyai.rl.PPOAlgoFlamingoHRL(envs, acmodel, args.discount, args.lr, args.beta1, args.beta2,
                                    args.gae_lambda, args.entropy_coef, args.value_loss_coef, args.max_grad_norm,
                                    args.optim_eps, args.clip_eps, args.ppo_epochs, obss_preprocessor,
                                    reshape_reward, agent=train_agent, num_episodes=args.num_episodes, use_subgoal_desc=args.use_subgoal_desc,
                                    num_episodes_per_batch=args.num_episodes_per_batch, use_FiLM=args.use_FiLM)

    # cases: train with recurrent observations
    # 1. not args.use_subgoal
    # 2. args.use_subgoal    
    else:
        algo = babyai.rl.PPOAlgo(envs, acmodel, args.frames_per_proc, args.discount, args.lr, args.beta1, args.beta2,
                                args.gae_lambda,args.entropy_coef, args.value_loss_coef, args.max_grad_norm, args.recurrence,
                                args.optim_eps, args.clip_eps, args.ppo_epochs, args.batch_size, obss_preprocessor,
                                reshape_reward, use_subgoal=args.use_subgoal, agent=train_agent)
else:
    raise ValueError("Incorrect algorithm name: {}".format(args.algo))

# When using extra binary information, more tensors (model params) are initialized compared to when we don't use that.
# Thus, there starts to be a difference in the random state. If we want to avoid it, in order to make sure that
# the results of supervised-loss-coef=0. and extra-binary-info=0 match, we need to reseed here.

utils.seed(args.seed)

# Restore training status

status_path = os.path.join(utils.get_log_dir(args.model), 'status.json')
if os.path.exists(status_path):
    with open(status_path, 'r') as src:
        status = json.load(src)
else:
    status = {'i': 0,
              'num_episodes': 0,
              'num_frames': 0}

# Define logger and Tensorboard writer and CSV writer
header = (["update", "episodes", "frames", "FPS", "duration"]
          + ["return_" + stat for stat in ['mean', 'std', 'min', 'max']]
          + ["success_rate"]
          + ["num_frames_" + stat for stat in ['mean', 'std', 'min', 'max']]
          + ["entropy", "value", "policy_loss", "value_loss", "loss", "grad_norm"])
# In the case of training a high-level policy that selects a low-level policy
# A low-level policy is supposed to solve a subtask or complete a subgoal, which
# is relevant to the initial mission gaol.
if args.use_subgoal:
    status['num_high_level_actions'] = 0 #status['num_primitive_steps'] = 0
    header = (header
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
    if args.use_subgoal:
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

        if args.use_subgoal:
            num_high_level_actions_per_episode = utils.synthesize(logs["num_high_level_actions_per_episode"])

        data = [status['i'], status['num_episodes'], status['num_frames'],
                fps, total_ellapsed_time,
                *return_per_episode.values(),
                success_per_episode['mean'],
                *num_frames_per_episode.values(),
                logs["entropy"], logs["value"], logs["policy_loss"], logs["value_loss"],
                logs["loss"], logs["grad_norm"]]

        format_str = ("U {} | E {} | F {:06} | FPS {:04.0f} | D {} | R:xsmM {: .2f} {: .2f} {: .2f} {: .2f} | "
                      "S {:.2f} | F:xsmM {:.1f} {:.1f} {} {} | H {:.3f} | V {:.3f} | "
                      "pL {: .3f} | vL {:.3f} | L {:.3f} | gN {:.3f} | ")

        if args.use_subgoal:
            data = data + [status["num_high_level_actions"], *num_high_level_actions_per_episode.values()]
            format_str = format_str + "hla {:06} | hla:xsmM {:.1f} {:.1f} {} {} | "

        logger.info(format_str.format(*data))
        if args.tb:
            assert len(header) == len(data)
            for key, value in zip(header, data):
                writer.add_scalar(key, float(value), status['num_frames'])

        csv_writer.writerow(data)

    # Save obss preprocessor vocabulary and model

    if args.save_interval > 0 and status['i'] % args.save_interval == 0:
        obss_preprocessor.vocab.save()
        with open(status_path, 'w') as dst:
            json.dump(status, dst)
            utils.save_model(acmodel, args.model)

        # Testing the model before saving
        if args.use_subgoal:
            agent = SkillModelAgent(
                args.model, obss_preprocessor, argmax=True,
                subgoals=None, goal=None, skill_library=skill_library, use_vlm=args.use_vlm, use_subgoal_desc=args.use_subgoal_desc)
        else:
            agent = ModelAgent(args.model, obss_preprocessor, argmax=True)

        if (not args.use_subgoal) and acmodel.use_vlm:
            history = acmodel.history
            acmodel.history = []
        
        agent.model = acmodel
        agent.model.eval()

        logs = batch_evaluate(
            agent,
            test_env_name,
            args.val_seed,
            args.val_episodes,
            pixel=use_pixel,
            concurrent_episodes=args.val_concurrent_episodes,
            use_subgoal=args.use_subgoal)

        agent.model.train()

        if (not args.use_subgoal) and acmodel.use_vlm:
            acmodel.history = history

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
            utils.save_model(acmodel, args.model + '_best')
            obss_preprocessor.vocab.save(utils.get_vocab_path(args.model + '_best'))
            logger.info("Return {: .2f}; best model is saved".format(mean_return))
        else:
            logger.info("Return {: .2f}; not the best model; not saved".format(mean_return))


print(f"Total time elapsed: {time.time() - total_start_time}")
