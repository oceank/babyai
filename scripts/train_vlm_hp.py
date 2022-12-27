#!/usr/bin/env python3

"""
Script to train a VLM as the high-level policy in a supervised way by using collected successful tracjectories.

Input Arguments:
    env: name of an environment level
    demos_name: filename (no file extension) of collected demonstrations
    abstract_history: an integer that indicates if a full history or an abstraction of a full history is used as the input of the VLM
        0: full history so far (vlm:only_attend_immediate_media = False)
        1: critical timesteps so far (vlm:only_attend_immediate_media = True)
    vlm_arc: name of the VLM
    lr: learning rate
"""

import os
import logging
import csv
import json
import gym
import datetime
import torch
import numpy as np
import argparse

from sklearn.model_selection import train_test_split
from torch.optim import AdamW

import babyai.utils as utils
from babyai.utils.format import RawImagePreprocessor
from babyai.utils.vlm import BowImageConvEncoder, train_test_helper, log_msg, calc_loss_per_subgoal
from babyai.levels.verifier import LowlevelInstrSet

from transformers import GPT2LMHeadModel, GPT2Tokenizer
from FlamingoGPT2.model import FlamingoGPT2


# Parse arguments
parser = argparse.ArgumentParser(
    prog = 'train_vlm_hp.py',
    description = 'Train a vision-language model',)

parser.add_argument("--env", default=None,
                            help="name of the environment to train on (REQUIRED)")
parser.add_argument("--demos_name", default=None,
                    help="demos filename (REQUIRED)")

parser.add_argument("--vlm_arc", default='Flamingo',
                    help="the architecture of the vision-language model (default: Flamingo)")
parser.add_argument("--max-history-window-vlm", type=int, default=128,
                    help="maximum number of observations that can be hosted in the history for VLM (default: 128)")
parser.add_argument("--lr", type=float, default=1e-5,
                    help="learning rate (default: 1e-5)")
parser.add_argument("--max-grad-norm", type=float, default=1.0,
                    help="global clipping norm (default: 1.0)")
parser.add_argument("--epochs", type=int, default=4,
                    help="number of epochs for training (default: 4)")
parser.add_argument("--abstract-history", action="store_true", default=False,
                    help="Allows you to switch between the full history and the abstraction of the full history")
parser.add_argument("--log-interval", type=int, default=10,
                    help="number of used demonstrations between two logging events during training (default: 10, 0 means no saving)")


args = parser.parse_args()

# Load the demonstrations and split it into training, validation and testing partitions
# "--env", "BabyAI-UnlockLocalR2Dist-v0",
# "--demos_name", "UnlockLocalR2Dist_BotDemosfrom babyai.levels.verifier import LowlevelInstrSet_100000",
model_name_prefix = args.demos_name
experiment_datetime = datetime.datetime.now().strftime("%y-%m-%d-%H-%M-%S")
if args.abstract_history:
    model_name_prefix += "_abstract_" + experiment_datetime
else:
    model_name_prefix += "_full_" + experiment_datetime

demos_train_set_path = os.path.join(utils.storage_dir(), "trainset.pkl") 
demos_test_set_path = os.path.join(utils.storage_dir(), "testset.pkl")

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
    demos_train ,demos_test = train_test_split(demos,test_size=test_samples_ratio)
    utils.save_demos(demos_train, demos_train_set_path)
    utils.save_demos(demos_test, demos_test_set_path)

model_dir = os.path.join(utils.storage_dir(), "models", model_name_prefix)
os.makedirs(model_dir)
training_status_path = os.path.join(model_dir, "training_status.txt")
vlm_model_path = os.path.join(model_dir, "vlm.pt")
image_conv_model_path = os.path.join(model_dir, "image_conv.pt")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

vlm = None
# Create the VLM with randomly initialized parameters
if args.vlm_arc == "Flamingo":
    lang_model_name = "distilgpt2" # "gpt2"
    # gpt2: 12 transformer layers
    # distilgpt2: 6 transformer layers

    msg = f"=== Initialize a visual-language model, FlamingoGPT2, to assist the agent ===" + "\n"
    msg += f"[Setup] Use VLM to help the anget to explore the grid world" + "\n"
    msg += f"[Setup] Create a tokenizer and {lang_model_name} language model"
    log_msg(training_status_path, msg)

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

    vit = None

    msg = f"[Setup] Create a Flamingo Model"
    log_msg(training_status_path, msg)
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
        perceiver_num_time_embeds = args.max_history_window_vlm,#16, 8, 128
        only_attend_immediate_media=args.abstract_history # True: Abstracted history, False: Full history
    )
else:
    raise("Unsupported VLM")

env = gym.make(args.env)
image_preproc = RawImagePreprocessor()
visual_observation_bow_flat_dim=147
bow_image_conv_encoder = BowImageConvEncoder(
    visual_observation_bow_flat_dim, vlm.wte.embedding_dim,image_preproc,device)
lowlevel_instr_set = LowlevelInstrSet()
vlm.to(device)

parameters = list(vlm.parameters()) + list(bow_image_conv_encoder.parameters())
optimizer = AdamW(parameters, lr=args.lr) # default lr is 5e-5

epoch_train_losses = np.zeros((args.epochs, 4))
epoch_test_losses = np.zeros((args.epochs, 4))
train_loss_path = os.path.join(model_dir, "train_loss.npy") 
test_loss_path = os.path.join(model_dir, "test_loss.npy")

msg = f"Training and testing start..."
log_msg(training_status_path, msg)

# Test if the two implemnetations of loss calculation over a trajectory are correct
unit_test = True
if unit_test:
    test_demo_idx = 100
    loss_per_csg_calc = calc_loss_per_subgoal(
        device, demos_train[test_demo_idx], args.abstract_history,
        lowlevel_instr_set, tokenizer, vlm, bow_image_conv_encoder,
        debug=True)
    loss_per_csg_calc = loss_per_csg_calc.item()

    is_training = False
    epoch_i = 0
    log_interval = 1
    loss_over_demo = train_test_helper(
        device, is_training, training_status_path, epoch_i,
        demos_train,
        log_interval, args.abstract_history,
        lowlevel_instr_set, tokenizer, vlm, bow_image_conv_encoder,
        optimizer, args.max_grad_norm,
        debug = True, test_demo_idx=test_demo_idx)
    print("[Unit Test: Summary")
    print(f"loss_per_csg_calc:{loss_per_csg_calc}")
    print(f"loss_over_demo   :{loss_over_demo}")
    print(f"Difference       :{round(loss_per_csg_calc-loss_over_demo, 6)}")

for epoch_i in range(0, args.epochs):
    msg = '======== Epoch {:} / {:} ========'.format(epoch_i + 1, args.epochs)
    log_msg(training_status_path, msg)

    # Training
    is_training = True
    tr_losses_stat = train_test_helper(
        device,
        is_training,
        training_status_path,
        epoch_i,
        demos_train,
        args.log_interval,
        args.abstract_history,
        lowlevel_instr_set,
        tokenizer,
        vlm,
        bow_image_conv_encoder,
        optimizer,
        args.max_grad_norm)

    epoch_train_losses[epoch_i] = tr_losses_stat
    np.save(train_loss_path, epoch_train_losses)
    torch.save(bow_image_conv_encoder, image_conv_model_path)
    torch.save(vlm, vlm_model_path)

    # Testing
    is_training = False
    te_losses_stat = train_test_helper(
        device,
        is_training,
        training_status_path,
        epoch_i,
        demos_test,
        args.log_interval,
        args.abstract_history,
        lowlevel_instr_set,
        tokenizer,
        vlm,
        bow_image_conv_encoder)

    epoch_test_losses[epoch_i] = te_losses_stat
    np.save(test_loss_path, epoch_test_losses)