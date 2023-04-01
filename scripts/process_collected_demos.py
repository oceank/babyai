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

from babyai.utils.model import load_model, save_model
from babyai.levels.verifier import LowlevelInstrSet, SKILL_DESCRIPTIONS, SKILL_DESC_INDICES
from sklearn.model_selection import train_test_split
import pandas as pd

#SKILL_DESCRIPTIONS = ["OpenDoor", "PassDoor", "OpenBox", "Pickup", "GoTo", "DropNextTo", "DropNextNothing"]

# Parse arguments
parser = ArgumentParser()
parser.add_argument("--demos-name-list", type=str, default=None,
                    help="name list of demos")
parser.add_argument("--demos-name", default=None,
                    help="demos filename (REQUIRED)")
parser.add_argument("--dataset-split-seed", type=int, default=1,
                    help="the seed used by train_test_split() to split the dataset"
                    )

args = parser.parse_args()
device = "cuda" if torch.cuda.is_available() else "cpu"
utils.seed(args.seed)

# Initialize subgoal set
print(f"===>    Initializing the predefined subgoal set")
subgoal_set = LowlevelInstrSet()
subgoal_set.display_all_subgoals()

demos_name_list = [
    "GoTo_BotDemos_1000000",
    "UnlockPickupDist_BotDemos_1000000",
    "UnlockLocalR2Dist_BotDemos_1000000",
    "PutNextLocal_BotDemos_1000000",
    "GoToObjMaze_BotDemos_1000000",
    "Open_BotDemos_1000000",
    "Pickup_BotDemos_1000000",
    "Unlock_BotDemos_1000000",
    "GoToSeq_BotDemos_1000000",
    "PutNext_BotDemos_1000000",
    "UnblockPickup_BotDemos_1000000",
    "BlockedUnlockPickup_BotDemos_1000000",
    "SynthLoc_BotDemos_1000000",
    "Synth_BotDemos_1000000",
    "SynthSeq_BotDemos_1000000",
    "GoToImpUnlock_BotDemos_1000000",
    "BossLevel_BotDemos_1000000"
]
row_indices = demos_name_list.copy()
row_indices.append('total')
applied_skill_counts = {}
for skill_desc in SKILL_DESCRIPTIONS:
    applied_skill_counts[skill_desc] = [0]*len(row_indices)
applied_skill_counts["total"] = [0]*len(row_indices)
df = pd.DataFrame(applied_skill_counts, index=row_indices, dtype=int)

demos_dir = os.path.join(utils.storage_dir(), "demos")
for demos_name in demos_name_list:
    print(f"Loading demos: {demos_name}")
    demos_train_set_path = os.path.join(demos_dir, demos_name+".pkl")
    demos = utils.load_demos(demos_train_set_path)
    print(f"Counting completed subgoals in loaded demos")
    demo_count = 0
    for demo in demos:
        demo_count += 1
        completed_subgaols = demo[4]
        for csg_indices in completed_subgaols:
            for csg_idx in csg_indices:
                skill_desc = subgoal_set.all_subgoals[csg_idx][2]
                df.loc[demos_name, skill_desc] += 1
        if demo_count%1000 == 0:
            print(df)
    df.loc[demos_name, 'total'] = df.loc[demos_name].sum()
    print(df)
    df.to_csv("applied_skills_demos_stat.csg")

df.loc['total', :] = df.sum(axis=0)
print(df)
df.to_csv("applied_skills_demos_stat.csv")
    



'''
demos_dir = os.path.join(utils.storage_dir(), "demos")
demos_train_set_path = os.path.join(demos_dir, args.demos_name+f"_dss{args.dataset_split_seed}_trainset.pkl")
demos_test_set_path = os.path.join(demos_dir, args.demos_name+f"_dss{args.dataset_split_seed}_testset.pkl")

if os.path.exists(demos_train_set_path) and os.path.exists(demos_test_set_path):
    demos_train = utils.load_demos(demos_train_set_path)
    demos_test = utils.load_demos(demos_test_set_path)
else:
    demos_path = utils.get_demos_path(args.demos_name, args.env, origin=None, valid=False)
    demos = utils.load_demos(demos_path)
    # demos: list of tuples
    #   tuple: (obs, action, done, completed_subgoals, reward, seed)
    #       obs: {'image':, 'direction':, 'mission':}
    #       action: an integer
    #       done: true or false
    #       completed_subgoals: [list of completed subgoals' indices at timestep t]
    #       reward: a real number between 0 and 1
    #       seed: an integer
    demos = utils.demos.transform_demos(demos, check_subgoal_completion=True)
    test_samples_ratio = 0.2 # episode-wise
    total_demos = len(demos)
    demos_train ,demos_test = train_test_split(demos,test_size=test_samples_ratio, random_state=args.dataset_split_seed)
    utils.save_demos(demos_train, demos_train_set_path)
    utils.save_demos(demos_test, demos_test_set_path)

'''







