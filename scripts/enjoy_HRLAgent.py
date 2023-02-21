#!/usr/bin/env python3

"""
Visualize the performance of a model on a given environment.
"""
import os
import argparse
import gym
import time


import babyai.utils as utils
from babyai.utils.model import create_random_hrl_vlm_model
from babyai.levels.verifier import LowlevelInstrSet

import numpy as np
import torch

# Parse arguments

parser = argparse.ArgumentParser()
parser.add_argument("--env", required=True,
                    help="name of the environment to be run (REQUIRED)")
parser.add_argument("--model", default=None,
                    help="name of the trained model (REQUIRED or --demos-origin or --demos REQUIRED) or SubGoalModelAgent")
parser.add_argument("--demos", default=None,
                    help="demos filename (REQUIRED or --model demos-origin required)")
parser.add_argument("--demos-origin", default=None,
                    help="origin of the demonstrations: human | agent (REQUIRED or --model or --demos REQUIRED)")
parser.add_argument("--seed", type=int, default=None,
                    help="random seed (default: 0 if model agent, 1 if demo agent)")
parser.add_argument("--argmax", action="store_true", default=False,
                    help="action with highest probability is selected for model agent")
parser.add_argument("--pause", type=float, default=0.1,
                    help="the pause between two consequent actions of an agent")
parser.add_argument("--manual-mode", action="store_true", default=False,
                    help="Allows you to take control of the agent at any point of time")

parser.add_argument("--max-history-window-vlm", type=int, default=16,
                    help="maximum number of observations that can be hosted in the history for VLM (default: 16)")
parser.add_argument("--skill-names-file", type=str, default="skill_model_names.txt",
                    help="File containing the names of the skills to be used for the mission")
parser.add_argument("--instr-arch", default="gru",
                    help="arch to encode instructions, possible values: gru, bigru, conv, bow (default: gru)")
parser.add_argument("--skill-arch", default='bow_endpool_res',
                    help="image embedding architecture")
parser.add_argument("--skill-budget-steps", type=int, default=24,
                    help="the maximum number of steps allowed for each skill (default: 24)")

parser.add_argument("--print-primitive-action-info", action="store_true", default=False,
                    help="print out the information of each primitive action by a skill")
parser.add_argument("--manuall-select-subgoal", action="store_true", default=False,
                    help="manually select a subgoal by an expert")

args = parser.parse_args()

# action map of primitive actions
action_map = {
    "left"      : "left",
    "right"     : "right",
    "up"        : "forward",
    "p"         : "pickup",
    "pageup"    : "pickup",
    "d"         : "drop",
    "pagedown"  : "drop",
    " "         : "toggle"
}

device = "cuda" if torch.cuda.is_available() else "cpu"

if args.seed is None:
    args.seed = 0
# Set seed for all randomness sources
utils.seed(args.seed)

# Load skill library
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
subgoal_set = LowlevelInstrSet()
subgoal_indices_str = [str(sidx) for sidx in range(subgoal_set.num_subgoals_info['total'])]
subgoal_set.display_all_subgoals()

# Create a random HRL-VLM model as the high-level policy if it does not exist
if args.model is None:
    num_high_level_actions = subgoal_set.num_subgoals_info['total']
    acmodel, args.model = create_random_hrl_vlm_model(
        args.env, args.seed, num_high_level_actions,
        args.skill_arch, args.instr_arch, args.max_history_window_vlm, device)

    path = utils.model.get_model_path(args.model)
    utils.create_folders_if_necessary(path)
    torch.save(acmodel, path)

max_num_episodes = 100
subgoal_idx = 0
step = 0
episode_num = 0
all_rewards = []

# Generate environment
env = gym.make(args.env)
env.seed(args.seed)

global obs
obs = env.reset()
mission = obs["mission"]

# Define and reset the agent
agent = utils.load_agent(
        env=env, model_name=args.model, argmax=args.argmax,
        skill_library=skill_library, skill_memory_size=skill_memory_size,
        subgoal_set=subgoal_set, use_vlm=True,)
agent.on_reset(env, mission, obs, propose_first_subgoal=(not args.manuall_select_subgoal))
print(f"[Episode: {episode_num+1}] Mission: {mission}")
if not args.manuall_select_subgoal:
    subgoal_idx += 1
    print(f"The {subgoal_idx}th subgoal is: {agent.current_subgoal_desc}")

# Run the agent
done = True
action = None

def get_statistics(arr, num_decimals=4):
    mean = np.round(arr.mean(), decimals=num_decimals)
    std = np.round(arr.std(), decimals=num_decimals)
    max = np.round(arr.max(), decimals=num_decimals)
    min = np.round(arr.min(), decimals=num_decimals)

    return mean, std, max, min

keyboard_input=""
def keyDownCb(event):
    global obs
    global step
    global episode_num
    global expected_completed_subgoals
    global subgoal_indices
    global keyboard_input
    global subgoal_idx
    global mission

    keyName = event.key
    while keyName != "enter":
        keyboard_input += keyName
        return
    if keyboard_input == "":
        keyboard_input = "enter"

    # Avoiding processing of observation by agent for wrong key clicks
    if not ((keyboard_input in action_map) or (keyboard_input == "enter") or (keyboard_input in subgoal_indices_str)):
        print(f"Keyboard input, {keyboard_input}, is not supported. Please try a valid one.")
        return


    # Map the key to an action
    if keyboard_input in action_map:
        action = env.actions[action_map[keyboard_input]]
    elif keyboard_input in subgoal_indices_str:
        subgoal_idx += 1
        agent.setup_new_subgoal_and_skill(env, int(keyboard_input))
        print(f"The {subgoal_idx}th subgoal is: [{agent.current_subgoal_idx}] {agent.current_subgoal_desc}")
        keyboard_input = ""
        return
    # Enter: executes the agent's action by the current skill
    elif keyboard_input == "enter":
        print(f"step: {step}, mission: {mission}")
        result = agent.act(obs)
        action = result['action'].item()
    # unsupported key
    else:
        print(f"The entered key, {keyboard_input}, is not supported.")
        return

    # reset keyboard_input
    keyboard_input = ""
    obs, reward, done, _ = env.step(action)

    if args.print_primitive_action_info:
        msg = ""
        if 'dist' in result and 'value' in result:
            dist, value = result['dist'], result['value']
            dist_str = ", ".join("{:.4f}".format(float(p)) for p in dist.probs[0])
            msg = "\tcurrent subgoal: {}, action: {}, dist: {}, entropy: {:.2f}, value: {:.2f}".format(
                agent.current_subgoal_desc, env.get_action_name(action), dist_str, float(dist.entropy()), float(value))
            #msg = "step: {}, mission: {}, action: {}, dist: {}, entropy: {:.2f}, value: {:.2f}".format(
            #    step, obs["mission"], env.get_action_name(action), dist_str, float(dist.entropy()), float(value))
        else:
            msg = "step: {}, mission: {}, action: {}".format(
                step, obs['mission'], env.get_action_name(action))
        print(msg)

    # Update the current_time_step and accumulate information to the agent's history
    agent.current_time_step += 1
    agent.accumulate_env_info_to_history(action, obs, reward, done)

    # check if the current subgoal is done
    agent.verify_current_subgoal(action)
    if agent.current_subgoal_status != 0:
        subgoal_success = agent.current_subgoal_status == 1
        subgoal_status_str = "Success" if subgoal_success else "Failure"
        print(f"[Step {step}] Subgoal {agent.current_subgoal_idx}: {subgoal_status_str}")

        # append the subgoal status to the agent's history
        agent.update_history_with_subgoal_status()

    if done:
        if agent.current_subgoal_status == 0:
            print(f"[Step {step}] Subgoal {agent.current_subgoal_idx}: Incomplete. But the mission is done.")
            agent.update_history_with_subgoal_status()

        print(f"Reward: {reward}\n")
        env.render("human")
        episode_num += 1
        env.seed(args.seed + episode_num)
        obs = env.reset()
        mission = obs['mission']
        agent.on_reset(env, mission, obs, propose_first_subgoal=(not args.manuall_select_subgoal))

        step = 0
        subgoal_idx = 0
        print(f"[Episode: {episode_num+1}] Mission: {obs['mission']}")
        if not args.manuall_select_subgoal:
            subgoal_idx += 1
            print(f"The {subgoal_idx}th subgoal is: {agent.current_subgoal_desc}")
    else:
        # the mission is done yet
        if agent.current_subgoal_status != 0 and (not args.manuall_select_subgoal):
            subgoal_idx += 1
            # the current subgoal is done, so propose the next subgoal
            agent.propose_new_subgoal(env)
            print(f"The {subgoal_idx}th subgoal is: [{agent.current_subgoal_idx}] {agent.current_subgoal_desc}")
        step += 1



if args.manual_mode:
    env.render('human')
    env.window.reg_key_handler(keyDownCb)

while True:
    time.sleep(args.pause)
    env.render("human")
    if not args.manual_mode:
        result = agent.act(obs)
        action = result['action'].item()
        obs, reward, done, _ = env.step(action)

        action_name = env.get_action_name(action)

        # agent.analyze_feedback(reward, done)
        if args.print_primitive_action_info:
            msg = ""
            if 'dist' in result and 'value' in result:
                dist, value = result['dist'], result['value']
                dist_str = ", ".join("{:.4f}".format(float(p)) for p in dist.probs[0])
                msg = "step: {}, mission: {}, action: {}, dist: {}, entropy: {:.2f}, value: {:.2f}".format(
                    step, obs["mission"], env.get_action_name(action), dist_str, float(dist.entropy()), float(value))
            else:
                msg = "step: {}, mission: {}, action: {}".format(
                    step, obs['mission'], env.get_action_name(action))
            print(msg)

        # Update the current_time_step and accumulate information to the agent's history
        agent.current_time_step += 1
        agent.accumulate_env_info_to_history(action, obs, reward, done)

        # check if the current subgoal is done
        agent.verify_current_subgoal(action)
        if agent.current_subgoal_status != 0:
            subgoal_success = agent.current_subgoal_status == 1
            subgoal_status_str = "Success" if subgoal_success else "Failure"
            print(f"[Step {step}] Subgoal {agent.current_subgoal_idx}: {subgoal_status_str}")

            # append the subgoal status to the agent's history
            agent.update_history_with_subgoal_status()

        if done:
            if agent.current_subgoal_status == 0:
                print(f"[Step {step}] Subgoal {agent.current_subgoal_idx}: Incomplete. But the mission is done.")
                agent.update_history_with_subgoal_status()

            print(f"Reward: {reward}\n")
            env.render("human")
            all_rewards.append(reward)
            episode_num += 1
            if episode_num == max_num_episodes:
                all_rewards = np.array(all_rewards)
                mean_r, std_r, max_r, min_r = get_statistics(all_rewards)
                print(f"Test Performance Over {max_num_episodes} Episodes: (Reward)")
                print(f"\tmean:{mean_r}\n")
                print(f"\tstd :{std_r}\n")
                print(f"\tmax :{max_r}\n")
                print(f"\tmin :{min_r}\n")
                break

            env.seed(args.seed + episode_num)
            obs = env.reset()
            agent.on_reset(env, obs['mission'], obs) # reset the history and propose the 1st subgoal

            step = 0
            print(f"[Episode: {episode_num+1}] Mission: {obs['mission']}")
            print(f"The new subgoal is: {agent.current_subgoal_desc}")
        else:
            # the mission is done yet
            if agent.current_subgoal_status != 0:
                # the current subgoal is done, so propose the next subgoal
                agent.propose_new_subgoal(env)
                print(f"The new subgoal is: {agent.current_subgoal_desc}")
            step += 1

    if env.window.closed:
        break
