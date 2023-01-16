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
import time
from functools import partial
import gym
import datetime
import torch
import numpy as np
import argparse

from sklearn.model_selection import train_test_split
from torch.optim import AdamW
from torch.utils.data import DataLoader

import babyai.utils as utils
from babyai.utils.format import RawImagePreprocessor
from babyai.utils.vlm import SubgoalsDemoDataset, subgoal_demo_collate_fn, SubgoalsDemoParsedDataset, subgoal_demo_parsed_collate_fn, SubgoalsDemoTokenizedDataset, subgoal_demo_tokenized_collate_fn
from babyai.utils.vlm import BowImageConvEncoder, log_msg, train_test_helper_batch_process, train_test_helper_batch_process_with_dataloader, format_time
from babyai.utils.vlm import num_unit_test_case, unit_test_loss_cal_by_subgoal_vs_episode, unit_test_loss_cal_by_episode_vs_batch
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
parser.add_argument("--batch-size", type=int, default=1,
                    help="number of episodes used during training between each model update (default: 1)")

parser.add_argument("--debug", action="store_true", default=False,
                    help="debug the implementation of two loss calculations")
parser.add_argument("--unit-test-case", type=int, default=0,
                    help=f"test case id. 0 indicates to run all test cases. Unit tests will only run when 'debug' is set." +
                        "\n\t1: test_loss_cal_by_subgoal_vs_episode" +
                        "\n\t2: test_loss_cal_by_episode_vs_batch"
                    )
parser.add_argument("--save-initial-model", action="store_true", default=False,
                    help="save the initial model to see if the model is learning anything at all")

parser.add_argument("--dataload-type", type=int, default=0,
                    help="Three types:"+
                        "\n\t0: demostration parsing (collate_fn), tokenization (collate_fn), padding (collate_fn)" +
                        "\n\t1: demostration parsing (init), tokenization (collate_fn), padding (collate_fn)" +
                        "\n\t2: demostration parsing (init), tokenization (init), padding (collate_fn)"
                    )
parser.add_argument("--pin-memory", action="store_true", default=False,
                    help="pin the loaded data to a memory block before moving to GPU")
parser.add_argument("--num-workers", type=int, default=0,
                    help="number of processer used to load data in Pytorch DataLoader"
                    )
   
args = parser.parse_args()

# Load the demonstrations and split it into training, validation and testing partitions
# "--env", "BabyAI-UnlockLocalR2Dist-v0",
# "--demos_name", "UnlockLocalR2Dist_BotDemosfrom babyai.levels.verifier import LowlevelInstrSet_100000",
model_name_prefix = args.demos_name + f"_b{args.batch_size}_lr{args.lr}"
experiment_datetime = datetime.datetime.now().strftime("%y-%m-%d-%H-%M-%S")
if args.abstract_history:
    model_name_prefix += "_abstract_" + experiment_datetime
else:
    model_name_prefix += "_full_" + experiment_datetime

demos_dir = os.path.join(utils.storage_dir(), "demos")
demos_train_set_path = os.path.join(demos_dir, args.demos_name+"_trainset.pkl") 
demos_test_set_path = os.path.join(demos_dir, args.demos_name+"_testset.pkl")

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
log_msg(training_status_path, f"Experiment Arguments: {args}\n")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ignored label (token) id that will be not considered when calculting the loss by the VLM
skip_label = -1

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
        only_attend_immediate_media=args.abstract_history, # True: Abstracted history, False: Full history
        skip_label = skip_label      # The label id that will be skipped when calculating loss
    )
else:
    raise("Unsupported VLM")

env = gym.make(args.env)
image_preproc = RawImagePreprocessor()
visual_observation_bow_flat_dim=147
bow_image_conv_encoder = BowImageConvEncoder(
    visual_observation_bow_flat_dim, vlm.wte.embedding_dim,
    None, #image_preproc,
    device)

vlm_model_path = os.path.join(model_dir, "vlm.pt")
image_conv_model_path = os.path.join(model_dir, "image_conv.pt")
vlm_model_path_best = os.path.join(model_dir, "vlm_best.pt")
image_conv_model_path_best = os.path.join(model_dir, "image_conv_best.pt")
if args.save_initial_model:
    vlm_model_path_init = os.path.join(model_dir, "vlm_init.pt")
    image_conv_model_path_init = os.path.join(model_dir, "image_conv_init.pt")
    torch.save(bow_image_conv_encoder, image_conv_model_path_init)
    torch.save(vlm, vlm_model_path_init)

lowlevel_instr_set = LowlevelInstrSet()
vlm.to(device)
bow_image_conv_encoder.to(device)

parameters = list(vlm.parameters()) + list(bow_image_conv_encoder.parameters())
optimizer = AdamW(parameters, lr=args.lr) # default lr is 5e-5

epoch_train_losses = np.zeros((args.epochs, 4))
epoch_test_losses = np.zeros((args.epochs, 4))
train_loss_path = os.path.join(model_dir, "train_loss.npy") 
test_loss_path = os.path.join(model_dir, "test_loss.npy")


msg = f"Creating dataloads..."
log_msg(training_status_path, msg)
tt = time.time()
pin_memory = args.pin_memory
num_workers = args.num_workers
if args.dataload_type == 0:
    # The collate function returns a tuple of four items: vlm_input, vlm_media, seeds, all_pre_csg_time_steps
    # vlm_input: Huggingface BatchEncoding object
    # vlm_media: image tensor with a shape, (b, t, ...)
    # seeds    : list of seeds
    # all_pre_csg_time_steps: list of lists of time steps in one episode when some subgoal completes.
    #                         The index of the last completes subgoal is exlucded since it is the last step of the episode.
    #                         The t=0 is included to facilite the processing.

    batch_collate_fn_partial = partial(
        subgoal_demo_collate_fn,
        abstract_history = args.abstract_history,
        lowlevel_instr_set = lowlevel_instr_set,
        tokenizer = tokenizer,
        image_preproc = image_preproc,
        skip_label = skip_label,
        pin_memory = pin_memory)
    train_dataset = SubgoalsDemoDataset(demos_train)
    test_dataset = SubgoalsDemoDataset(demos_test)
elif args.dataload_type == 1:
    batch_collate_fn_partial = partial(
        subgoal_demo_parsed_collate_fn,
        tokenizer = tokenizer,
        image_preproc = image_preproc,
        skip_label = skip_label,
        pin_memory = pin_memory)
    train_dataset = SubgoalsDemoParsedDataset(demos_train, args.abstract_history, lowlevel_instr_set)
    test_dataset = SubgoalsDemoParsedDataset(demos_test, args.abstract_history, lowlevel_instr_set)
elif args.dataload_type == 2:
    batch_collate_fn_partial = partial(
        subgoal_demo_tokenized_collate_fn,
        tokenizer = tokenizer,
        image_preproc = image_preproc,
        skip_label = skip_label,
        pin_memory = pin_memory)
    train_dataset = SubgoalsDemoTokenizedDataset(demos_train, args.abstract_history, lowlevel_instr_set, tokenizer, skip_label)
    test_dataset = SubgoalsDemoTokenizedDataset(demos_test, args.abstract_history, lowlevel_instr_set, tokenizer, skip_label)

train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=batch_collate_fn_partial, num_workers=num_workers)
test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=batch_collate_fn_partial, num_workers=num_workers)
msg = f"Time cost of creating data loads: {format_time(time.time() - tt)}"
log_msg(training_status_path, msg)

msg = f"Training and testing start..."
log_msg(training_status_path, msg)

'''
log_msg(training_status_path, "Parse all demonstrations: started")
t0 = time.time()
device_cpu = torch.device("cpu")
train_dataset = parse_collected_demos(device_cpu, demos_train, args.abstract_history, lowlevel_instr_set, tokenizer, skip_label=-1, max_token_seq_len = 512)
train_dataset2 = parse_collected_demos_per_demo(device, demos_train, args.abstract_history, lowlevel_instr_set, tokenizer, skip_label=-1)
test_dataset = parse_collected_demos(device_cpu, demos_test, args.abstract_history, lowlevel_instr_set, tokenizer, skip_label=-1, max_token_seq_len = 512)
time_elapse = format_time(time.time() - t0)
log_msg(training_status_path, f"Parse all demonstrations: finished - {time_elapse}")
'''

if args.debug:
    demos = demos_test
    if args.unit_test_case==0:
        test_cases = range(1, num_unit_test_case+1)
    else:
        test_cases = [args.unit_test_case]
    
    for test_case_id in test_cases:
        if test_case_id == 1:
            test_demo_idx = 100
            test_demo = demos[test_demo_idx]
            unit_test_loss_cal_by_subgoal_vs_episode(
                test_demo, device, args.abstract_history,
                lowlevel_instr_set, tokenizer, vlm, bow_image_conv_encoder,
                skip_label)
        elif test_case_id == 2:
            test_demo_ids = [0, 1]
            test_demos = [demos[id] for id in test_demo_ids]
            unit_test_loss_cal_by_episode_vs_batch(
                test_demos, device, args.abstract_history,
                lowlevel_instr_set, tokenizer, vlm, bow_image_conv_encoder,
                skip_label
            )
else:
    best_test_loss = np.inf
    for epoch_id in range(0, args.epochs):
        msg = '======== Epoch {:} / {:} ========'.format(epoch_id + 1, args.epochs)
        log_msg(training_status_path, msg)

        # Training
        is_training = True
        '''
        tr_losses_stat = train_test_helper(
            device,
            is_training,
            training_status_path,
            epoch_id,
            demos_train,
            args.log_interval,
            args.abstract_history,
            lowlevel_instr_set,
            tokenizer,
            vlm,
            bow_image_conv_encoder,
            skip_label=skip_label,
            optimizer=optimizer,
            max_grad_norm=args.max_grad_norm,
            batch_size=args.batch_size)

        tr_losses_stat = train_test_helper_batch_process(
            device,
            is_training,
            training_status_path,
            epoch_id,
            train_dataset,
            args.log_interval,
            vlm,
            bow_image_conv_encoder,
            optimizer=optimizer,
            max_grad_norm=args.max_grad_norm,
            batch_size=args.batch_size)
        '''
        num_all_train_demos = len(demos_train)
        tr_losses_stat = train_test_helper_batch_process_with_dataloader(
            device,
            is_training,
            training_status_path,
            epoch_id,
            train_loader,
            num_all_train_demos,
            args.log_interval,
            vlm,
            bow_image_conv_encoder,
            optimizer=optimizer,
            max_grad_norm=args.max_grad_norm)

        epoch_train_losses[epoch_id] = tr_losses_stat
        np.save(train_loss_path, epoch_train_losses)
        # saved the trained model after each epoch
        torch.save(bow_image_conv_encoder, image_conv_model_path)
        torch.save(vlm, vlm_model_path)

        # Testing
        is_training = False
        '''
        te_losses_stat = train_test_helper(
            device,
            is_training,
            training_status_path,
            epoch_id,
            demos_test,
            args.log_interval,
            args.abstract_history,
            lowlevel_instr_set,
            tokenizer,
            vlm,
            bow_image_conv_encoder,
            skip_label=skip_label,
            batch_size=args.batch_size)

        te_losses_stat = train_test_helper_batch_process(
            device,
            is_training,
            training_status_path,
            epoch_id,
            test_dataset,
            args.log_interval,
            vlm,
            bow_image_conv_encoder,
            batch_size=args.batch_size)
        '''
        num_all_test_demos = len(demos_test)
        te_losses_stat = train_test_helper_batch_process_with_dataloader(
            device,
            is_training,
            training_status_path,
            epoch_id,
            test_loader,
            num_all_test_demos,
            args.log_interval,
            vlm,
            bow_image_conv_encoder)


        epoch_test_losses[epoch_id] = te_losses_stat
        np.save(test_loss_path, epoch_test_losses)

        if best_test_loss > te_losses_stat[0]:
            best_test_loss = te_losses_stat[0]
            torch.save(bow_image_conv_encoder, image_conv_model_path_best)
            torch.save(vlm, vlm_model_path_best)