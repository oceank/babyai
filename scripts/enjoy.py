#!/usr/bin/env python3

"""
Visualize the performance of a model on a given environment.
"""

import argparse
import gym
import time

import babyai.utils as utils
from babyai.levels.verifier import *

import numpy as np

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

parser.add_argument("--subgoal-model-name-list", type=str, default=None,
                    help="list of model name of each subgoal")

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

# skill map: each skill is supposed to complete a low-level 
skill_map = {
    "0" : 0, # PickupKeyLocal
    "1" : 1, # OpenDoorLocal
}

use_subgoals = False
if args.model=="SubGoalModelAgent":
   assert args.subgoal_model_name_list is not None
   use_subgoals = True
   subgoal_model_name_list = args.subgoal_model_name_list.split(',')

assert args.model is not None or args.demos is not None, "--model or --demos must be specified."
if args.seed is None:
    args.seed = 0 if args.model is not None else 1

max_num_episodes = 100
all_rewards = []

# Set seed for all randomness sources

utils.seed(args.seed)

# Generate environment

env = gym.make(args.env)
env.seed(args.seed)

global obs
obs = env.reset()
print("Mission: {}".format(obs["mission"]))

# If subgoals and their policy models are used to commplete the mission
goal = None
subgoals = None
if use_subgoals:
    goal = {'desc':env.mission, 'instr':env.instrs}

    print(f"List of subgoals for the mission:")
    subgoals = env.sub_goals
    for subgoal, subgoal_model_name in zip(subgoals, subgoal_model_name_list):
        subgoal['model_name'] = subgoal_model_name
        print(f"*** Subgoal: {subgoal['desc']}")


# Define agent
agent = utils.load_agent(env, args.model, args.demos, args.demos_origin, args.argmax, args.env, subgoals, goal)
if args.demos is not None:
    max_num_episodes = len(agent.demos)
# Run the agent

done = True

action = None

# List of object colors
COLORS = [
    'red'   ,
    'green' ,
    'blue'  ,
    'purple',
    'yellow',
    'grey'  ,
]

# Map of object type to integers
OBJECT_TYPES = [
    'door'          ,
    'key'           ,
    'ball'          ,
    'box'           ,
]

subgoal_instructions = []
open_instrs = []
pickup_instrs = []
goto_instrs = []
pass_instrs = []


# OpenDoor
# PassDoor
# OpenBox
# Pickup: box, ball, key
# DropNext: next to box, ball, key or door
# GoToBalll
for obj_type in OBJECT_TYPES:
    for color in COLORS:
        obj = ObjDesc(obj_type, color=color)
        if obj_type == 'door':
            open_instrs.append(OpenInstr(obj))
            pass_instrs.append(PassInstr(obj))
        else:
            pickup_instrs.append(PickupInstr(obj))
            if obj_type == 'box':
                open_instrs.append(OpenBoxInstr(obj))

        goto_instrs.append(GoToInstr(obj))


subgoal_instructions = open_instrs
subgoal_instructions.extend(pass_instrs)
subgoal_instructions.extend(goto_instrs)
subgoal_instructions.extend(pickup_instrs)

def filter_valid_subgoal_instrs(subgoal_instructions, env):
    valid_subgoal_instructions = []
    for instr in subgoal_instructions:
        instr.reset_verifier(env)
        if len(instr.desc.obj_set) > 0:
            instr.instr_desc = instr.surface(env)
            valid_subgoal_instructions.append(instr)

    return valid_subgoal_instructions

def check_completed_subgoals(initial_valid_subgoal_instructions, subgoal_instructions, action, env):
    msg = ""
    completed_subgoals = 0
    objects_picked_or_dropped = False
    for instruction in subgoal_instructions:
        result = instruction.verify(action)
        if result == 'success':
            completed_subgoals += 1
            msg += f"\tSG{completed_subgoals}: {instruction.instr_desc}\n"
            # When 'pickup' or 'drop' instruction succeeds, the grid is changed.
            # So, the valid subgoal instructions need to be updated
            if isinstance(instruction, PickupInstr):
                objects_picked_or_dropped = True
    if objects_picked_or_dropped:
        subgoal_instructions = filter_valid_subgoal_instrs(initial_valid_subgoal_instructions, env)
    return msg, subgoal_instructions

def get_statistics(arr, num_decimals=4):
    mean = np.round(arr.mean(), decimals=num_decimals)
    std = np.round(arr.std(), decimals=num_decimals)
    max = np.round(arr.max(), decimals=num_decimals)
    min = np.round(arr.min(), decimals=num_decimals)

    return mean, std, max, min

def keyDownCb(event):
    global obs
    global use_subgoals
    global step
    global initial_valid_subgoal_instructions
    global valid_subgoal_instructions
    global episode_num

    keyName = event.key

    # Avoiding processing of observation by agent for wrong key clicks
    if keyName not in skill_map and keyName not in action_map and keyName != "enter":
        print(f"Keyboard input, {keyName}, is not supported. Please try a valid one.")
        return

    # Select a skill to work on a subtask task
    if use_subgoals and (keyName in skill_map):
        new_low_level_task_idx = skill_map[keyName]
        agent.select_new_subgoal(new_low_level_task_idx)
        agent.current_subgoal_instr.reset_verifier(env)
        print(f"[Input {keyName}] select the skill,{agent.current_subgoal_idx}, to complete the subtask, {agent.current_subgoal_desc}")
 
        # workaround to fix the instruction in the observation when working on the subgoal
        obs["mission"] = agent.current_subgoal_desc

        return

    else: # apply the policy of selected subgoal

        # Map the key to an action
        if keyName in action_map:
            action = env.actions[action_map[keyName]]

        # Enter executes the agent's action
        elif keyName == "enter":
            if use_subgoals and (agent.current_subgoal_idx is None):
                print(f"[No subgoal is selected thus no policy is available for use. Please choose the next subgoal:]")
                return
            else:
                action = agent.act(obs)['action']

        # unsupported key
        else:
            print(f"The entered key, {keyName}, is not supported.")
            return

        obs, reward, done, _ = env.step(action)

        print(f"[Low-level Step {step}], mission: {obs['mission']}, action: {env.get_action_name(action)}")
        msg, valid_subgoal_instructions = check_completed_subgoals(initial_valid_subgoal_instructions, valid_subgoal_instructions, action, env)
        print(msg)

        step += 1

        #verify_current_subgoal_helper(use_subgoals, agent, action, env, obs)
        if use_subgoals:
            is_subgoal_completed = agent.verify_current_subgoal(action)
            # workaround to fix the instruction in the observation when working on the subgoal
            if not is_subgoal_completed:
                obs['mission'] = agent.current_subgoal_desc

        # only 'done' parameter is used to reset the agent's memory for the finished mission
        agent.analyze_feedback(reward, done)
        if done:
            print(f"Reward: {reward}\n")
            episode_num += 1
            env.seed(args.seed + episode_num)
            obs = env.reset()
            print("Mission: {}".format(obs["mission"]))
            step = 0

            if use_subgoals:
                agent.reinitialize_mission(env)

            initial_valid_subgoal_instructions = filter_valid_subgoal_instrs(subgoal_instructions, env)
            valid_subgoal_instructions = initial_valid_subgoal_instructions.copy()
            print(f"\n# of valid subgoals: {len(valid_subgoal_instructions)}")

        if use_subgoals and (done or is_subgoal_completed):
            print(f"[Please choose the next subgoal:]")



if args.manual_mode:
    env.render('human')
    env.window.reg_key_handler(keyDownCb)

if use_subgoals:
    print(f"[Please choose the next subgoal:]")

print(f"Total number of subgoals: {len(subgoal_instructions)}")
initial_valid_subgoal_instructions = filter_valid_subgoal_instrs(subgoal_instructions, env)
valid_subgoal_instructions = initial_valid_subgoal_instructions.copy()
print(f"# of valid subgoals: {len(valid_subgoal_instructions)}")

step = 0
episode_num = 0
while True:
    time.sleep(args.pause)
    env.render("human")
    if not args.manual_mode:
        result = agent.act(obs)
        action = result['action']
        obs, reward, done, _ = env.step(action)

        action_name = env.get_action_name(action)

        #verify_current_subgoal_helper(use_subgoals, agent, action, env, obs)
        if use_subgoals:
            is_subgoal_completed = agent.verify_current_subgoal(action)

        agent.analyze_feedback(reward, done)
        if 'dist' in result and 'value' in result:
            dist, value = result['dist'], result['value']
            dist_str = ", ".join("{:.4f}".format(float(p)) for p in dist.probs[0])
            print("step: {}, mission: {}, dist: {}, entropy: {:.2f}, value: {:.2f}".format(
                step, obs["mission"], dist_str, float(dist.entropy()), float(value)))
        else:
            print("step: {}, mission: {}, action: {}".format(step, obs['mission'], env.get_action_name(action)))
        
        msg, valid_subgoal_instructions = check_completed_subgoals(initial_valid_subgoal_instructions, valid_subgoal_instructions, action, env)
        print(msg)

        if done:
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

            print(f"[Episode: {episode_num+1}] Mission: {obs['mission']}")
            #reinitialize_mission_and_subgoals_helper(use_subgoals, agent, env, obs)
            if use_subgoals:
                agent.reinitialize_mission(env)
                print(f"[Please choose the next subgoal:]")

            agent.on_reset()
            step = 0

            initial_valid_subgoal_instructions = filter_valid_subgoal_instrs(subgoal_instructions, env)
            valid_subgoal_instructions = initial_valid_subgoal_instructions.copy()
            print(f"\n# of valid subgoals: {len(valid_subgoal_instructions)}")
        else:
            if use_subgoals and is_subgoal_completed:
                print(f"[Please choose the next subgoal:]")

            else:
                step += 1

    if env.window.closed:
        break
