#!/usr/bin/env python3

"""
Generate a set of agent demonstrations.

The agent can either be a trained model or the heuristic expert (bot).

Demonstration generation can take a long time, but it can be parallelized
if you have a cluster at your disposal. Provide a script that launches
make_agent_demos.py at your cluster as --job-script and the number of jobs as --jobs.


"""

import argparse
import gym
import logging
import sys
import subprocess
import os
import time
import numpy as np
import blosc
import torch

import babyai.utils as utils
from babyai.levels.verifier import LowlevelInstrSet

# Parse arguments

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--env", required=True,
                    help="name of the environment to be run (REQUIRED)")
parser.add_argument("--model", default='BOT',
                    help="name of the trained model (REQUIRED)")
parser.add_argument("--demos", default=None,
                    help="path to save demonstrations (based on --model and --origin by default)")
parser.add_argument("--episodes", type=int, default=1000,
                    help="number of episodes to generate demonstrations for")
parser.add_argument("--valid-episodes", type=int, default=512,
                    help="number of validation episodes to generate demonstrations for")
parser.add_argument("--seed", type=int, default=0,
                    help="start random seed")
parser.add_argument("--argmax", action="store_true", default=False,
                    help="action with highest probability is selected")
parser.add_argument("--log-interval", type=int, default=100,
                    help="interval between progress reports")
parser.add_argument("--save-interval", type=int, default=10000,
                    help="interval between demonstrations saving")
parser.add_argument("--filter-steps", type=int, default=0,
                    help="filter out demos with number of steps more than filter-steps")
parser.add_argument("--on-exception", type=str, default='warn', choices=('warn', 'crash'),
                    help="How to handle exceptions during demo generation")

parser.add_argument("--job-script", type=str, default=None,
                    help="The script that launches make_agent_demos.py at a cluster.")
parser.add_argument("--jobs", type=int, default=0,
                    help="Split generation in that many jobs")

args = parser.parse_args()
logger = logging.getLogger(__name__)

# Set seed for all randomness sources


def print_demo_lengths(demos):
    num_frames_per_episode = [len(demo[2]) for demo in demos]
    logger.info('Demo length: {:.3f}+-{:.3f}'.format(
        np.mean(num_frames_per_episode), np.std(num_frames_per_episode)))


def generate_demos(n_episodes, valid, seed, shift=0):
    utils.seed(seed)

    # Initialize the set of instructions for low-level tasks
    lowlevel_instr_set = LowlevelInstrSet()
    # print(f"Total number of subgoals: {len(lowlevel_instr_set.all_subgoals)}")

    # Generate environment
    env = gym.make(args.env)

    agent = utils.load_agent(env, args.model, argmax=args.argmax, demos_name=args.demos, demos_origin='agent', env_name=args.env, model_version="best")
    demos_path = utils.get_demos_path(args.demos, args.env, 'agent', valid)
    demos_status_path = demos_path[:-4] + "_status.txt"
    demos = []

    checkpoint_time = time.time()

    with open(demos_status_path, 'w') as f:
        f.write(f"{args.env}: Collection Started!\n")

    just_crashed = False
    # idx 0: count of time steps when no subgoal is completed
    # idx 1: count of time steps when one subgoal is completed
    # idx 2: count of time steps when >1 subgoals is completed
    completed_subgoals_counts = [0, 0, 0]
    while True:
        if len(demos) == n_episodes:
            break

        done = False
        if just_crashed:
            logger.info("reset the environment to find a mission that the bot can solve")
            env.reset()
        else:
            env.seed(seed + len(demos))
        obs = env.reset()
        agent.on_reset()

        lowlevel_instr_set.reset_valid_subgoals(env)
        # print(f"# of valid subgoals: {len(lowlevel_instr_set.current_valid_subgoals)}")

        actions = []
        mission = obs["mission"]
        images = []
        directions = []
        completed_subgoals = []

        try:
            while not done:
                action = agent.act(obs)['action']
                if isinstance(action, torch.Tensor):
                    action = action.item()
                new_obs, reward, done, _ = env.step(action)
                agent.analyze_feedback(reward, done)

                current_completed_subgoals = lowlevel_instr_set.check_completed_subgoals(action, env)

                actions.append(action)
                images.append(obs['image'])
                directions.append(obs['direction'])

                completed_subgoals.append(current_completed_subgoals)

                obs = new_obs

            if reward > 0 and (args.filter_steps == 0 or len(images) <= args.filter_steps):
                demos.append((mission, blosc.pack_array(np.array(images)), directions, actions, completed_subgoals, reward, seed+len(demos)))
                just_crashed = False

            if reward == 0:
                if args.on_exception == 'crash':
                    raise Exception("mission failed, the seed is {}".format(seed + len(demos)))
                just_crashed = True
                logger.info("mission failed")
        except (Exception, AssertionError):
            if args.on_exception == 'crash':
                raise
            just_crashed = True
            logger.exception("error while generating demo #{}".format(len(demos)))
            continue

        for time_step, csg in enumerate(completed_subgoals):
            if len(csg) == 0:
                completed_subgoals_counts[0] += 1
            elif len(csg) == 1:
                completed_subgoals_counts[1] += 1
            else: # len(csg) > 1
                completed_subgoals_counts[2] += 1
                msg_csg = lowlevel_instr_set.get_completed_subgoals_msg(csg)
                print(f"\t[demo #{len(demos) - 1}, t={time_step}] {msg_csg}")
    
        if len(demos) and len(demos) % args.log_interval == 0:
            
            now = time.time()
            demos_per_second = args.log_interval / (now - checkpoint_time)
            to_go = (n_episodes - len(demos)) / demos_per_second

            total_time_steps = sum(completed_subgoals_counts)
            csg0 = completed_subgoals_counts[0]/total_time_steps
            csg1 = completed_subgoals_counts[1]/total_time_steps
            csg2 = completed_subgoals_counts[2]/total_time_steps

            status_msg = "{}: demo #{}, {:.3f} demos per second, {:.3f} seconds to go, 0sg({:.3f}), 1sg({:.3f}), >1sg({:.3f})".format(
                args.env, len(demos) - 1, demos_per_second, to_go, csg0, csg1, csg2)
            logger.info(status_msg)
            with open(demos_status_path, 'a') as f:
                f.write(status_msg + "\n")
            checkpoint_time = now

        # Save demonstrations

        if args.save_interval > 0 and len(demos) < n_episodes and len(demos) % args.save_interval == 0:
            logger.info("Saving demos...")
            utils.save_demos(demos, demos_path)
            logger.info("{} demos saved".format(len(demos)))
            # print statistics for the last 100 demonstrations
            print_demo_lengths(demos[-100:])


    # Save demonstrations
    logger.info("Saving demos...")
    utils.save_demos(demos, demos_path)
    logger.info("{} demos saved".format(len(demos)))
    print_demo_lengths(demos[-100:])

    with open(demos_status_path, 'w') as f:
        f.write(f"{args.env}: Collection Done!\n")

def generate_demos_cluster():
    demos_per_job = args.episodes // args.jobs
    demos_path = utils.get_demos_path(args.demos, args.env, 'agent')
    job_demo_names = [os.path.realpath(demos_path + '.shard{}'.format(i))
                     for i in range(args.jobs)]
    for demo_name in job_demo_names:
        job_demos_path = utils.get_demos_path(demo_name)
        if os.path.exists(job_demos_path):
            os.remove(job_demos_path)

    command = [args.job_script]
    command += sys.argv[1:]
    for i in range(args.jobs):
        cmd_i = list(map(str,
            command
              + ['--seed', args.seed + i * demos_per_job]
              + ['--demos', job_demo_names[i]]
              + ['--episodes', demos_per_job]
              + ['--jobs', 0]
              + ['--valid-episodes', 0]))
        logger.info('LAUNCH COMMAND')
        logger.info(cmd_i)
        output = subprocess.check_output(cmd_i)
        logger.info('LAUNCH OUTPUT')
        logger.info(output.decode('utf-8'))

    job_demos = [None] * args.jobs
    while True:
        jobs_done = 0
        for i in range(args.jobs):
            if job_demos[i] is None or len(job_demos[i]) < demos_per_job:
                try:
                    logger.info("Trying to load shard {}".format(i))
                    job_demos[i] = utils.load_demos(utils.get_demos_path(job_demo_names[i]))
                    logger.info("{} demos ready in shard {}".format(
                        len(job_demos[i]), i))
                except Exception:
                    logger.exception("Failed to load the shard")
            if job_demos[i] and len(job_demos[i]) == demos_per_job:
                jobs_done += 1
        logger.info("{} out of {} shards done".format(jobs_done, args.jobs))
        if jobs_done == args.jobs:
            break
        logger.info("sleep for 60 seconds")
        time.sleep(60)

    # Training demos
    all_demos = []
    for demos in job_demos:
        all_demos.extend(demos)
    utils.save_demos(all_demos, demos_path)


logging.basicConfig(level='INFO', format="%(asctime)s: %(levelname)s: %(message)s")
logger.info(args)
# Training demos
if args.jobs == 0:
    generate_demos(args.episodes, False, args.seed)
else:
    generate_demos_cluster()
# Validation demos
if args.valid_episodes:
    generate_demos(args.valid_episodes, True, int(1e9))
