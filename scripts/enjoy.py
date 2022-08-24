#!/usr/bin/env python3

"""
Visualize the performance of a model on a given environment.
"""

import argparse
import gym
import time

import babyai.utils as utils

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

use_subgoals = False
if args.model=="SubGoalModelAgent":
   assert args.subgoal_model_name_list is not None
   use_subgoals = True
   subgoal_model_name_list = args.subgoal_model_name_list.split(',')

assert args.model is not None or args.demos is not None, "--model or --demos must be specified."
if args.seed is None:
    args.seed = 0 if args.model is not None else 1

# Set seed for all randomness sources

utils.seed(args.seed)

# Generate environment

env = gym.make(args.env)
env.seed(args.seed)

global obs
obs = env.reset()
print("Mission: {}".format(obs["mission"]))

# If subgoals and their policy models are used to commplete the mission
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

# Set the mission to be the first subgoal and reset its instruction's verifier
if use_subgoals:
    env.instrs = agent.current_subgoal_instr
    env.mission = agent.current_subgoal_desc
    env.instrs.reset_verifier(env)

# Run the agent

done = True

action = None

def keyDownCb(event):
    global obs
    global use_subgoals

    keyName = event.key
    print(keyName)

    # Avoiding processing of observation by agent for wrong key clicks
    if keyName not in action_map and keyName != "enter":
        return

    agent_action = agent.act(obs)['action']

    # Map the key to an action
    if keyName in action_map:
        action = env.actions[action_map[keyName]]

    # Enter executes the agent's action
    elif keyName == "enter":
        action = agent_action

    obs, reward, done, _ = env.step(action)

    if use_subgoals:
        status = agent.current_subgoal_instr.verify(action)
        if (status == 'success'):
            print(f"===> [Subgoal Completed] {agent.current_subgoal_desc}")
            if agent.current_subgoal_idx + 1 < len(agent.subgoals):
                new_subgoal_idx = agent.current_subgoal_idx + 1
                agent.select_new_subgoal(new_subgoal_idx)
                env.instrs = agent.current_subgoal_instr
                env.mission = agent.current_subgoal_desc
                env.instrs.reset_verifier(env)

                done = False # the initial goal is not completed yet
            else: # all subgoals have been completed
                env.instrs = agent.goal['instr']
                env.mission = agent.goal['desc']
                env.instrs.reset_verifier(env)
                if env.instrs.verify(action) == 'success': # the initial goal is completed
                    done = True
                    reward = env._reward()

    # only 'done' parameter is used to update the memory
    agent.analyze_feedback(reward, done)
    if done:
        print("Reward:", reward)
        obs = env.reset()
        print("Mission: {}".format(obs["mission"]))


if args.manual_mode:
    env.render('human')
    env.window.reg_key_handler(keyDownCb)

step = 0
episode_num = 0
while True:
    time.sleep(args.pause)
    env.render("human")
    if not args.manual_mode:
        result = agent.act(obs)
        obs, reward, done, _ = env.step(result['action'])
        agent.analyze_feedback(reward, done)
        if 'dist' in result and 'value' in result:
            dist, value = result['dist'], result['value']
            dist_str = ", ".join("{:.4f}".format(float(p)) for p in dist.probs[0])
            print("step: {}, mission: {}, dist: {}, entropy: {:.2f}, value: {:.2f}".format(
                step, obs["mission"], dist_str, float(dist.entropy()), float(value)))
        else:
            print("step: {}, mission: {}".format(step, obs['mission']))
        if done:
            print("Reward:", reward)
            episode_num += 1
            env.seed(args.seed + episode_num)
            obs = env.reset()
            agent.on_reset()
            step = 0
        else:
            step += 1

    if env.window.closed:
        break
