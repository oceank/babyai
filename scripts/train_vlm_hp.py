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
import time
import datetime
import torch
import numpy as np
import argparse
import gc

from einops import rearrange
import torch.nn as nn
from sklearn.model_selection import train_test_split


import babyai.utils as utils
from babyai.model import ImageBOWEmbedding, initialize_parameters
from babyai.utils.format import RawImagePreprocessor
from babyai.levels.verifier import LowlevelInstrSet

from transformers import GPT2LMHeadModel, GPT2Tokenizer, top_k_top_p_filtering, get_linear_schedule_with_warmup
from FlamingoGPT2.model import FlamingoGPT2
from FlamingoGPT2.utils import * # train, visualize_training_stats, load_processed_raw_data, prepare_dataload


class BowImageConvEncoder(nn.Module):
    def __init__(self, visual_observation_bow_flat_dim, embedding_dim, image_preproc, device):
        super().__init__()
        self.visual_observation_bow_flat_dim = visual_observation_bow_flat_dim
        self.embedding_dim = embedding_dim
        self.image_preproc = image_preproc
        self.device = device
        self.image_conv = nn.Sequential(*[
                    ImageBOWEmbedding(self.visual_observation_bow_flat_dim, self.embedding_dim),
                    nn.Conv2d(
                        in_channels=self.embedding_dim, out_channels=self.embedding_dim,
                        kernel_size=(3, 3), stride=1, padding=1),
                    nn.BatchNorm2d(self.embedding_dim),
                    nn.ReLU(),
                    nn.Conv2d(in_channels=self.embedding_dim, out_channels=self.embedding_dim, kernel_size=(3, 3), padding=1),
                    nn.BatchNorm2d(self.embedding_dim),
                    nn.ReLU(),
                ])
        self.image_conv.apply(initialize_parameters)
        self.image_conv.to(device)
    
    def forward(self, obss):
        batch_size = 1
        images = self.image_preproc(obss, device=self.device)
        images = images.unsqueeze(dim=0)
        images = rearrange(images, 'b t h w c -> (b t) c h w')
        image_embeds = self.image_conv(images)
        # convert to the shape : (batch_size, times, height*width, feature_dim)
        image_embeds = rearrange(image_embeds, '(b t) d h w-> b t (h w) d', b=batch_size)
        return image_embeds

def log_msg(fp, msg, logging_mode='a'):
    with open(fp, logging_mode) as f:
        f.write(msg + "\n")

def get_stat(arr, rd=4, return_type='plain'):
    ave = round(np.mean(arr),rd)
    std = round(np.std(arr), rd)
    max = round(np.max(arr), rd)
    min = round(np.min(arr), rd)
    if return_type == "np":
        return np.array([ave, std, max, min])
    else:
        return ave, std, max, min

def log_losses_stat(fp, losses, t0, epoch_id, is_training):
    avg_loss, std_loss, max_loss, min_loss = get_stat(losses)
    time_elapse = format_time(time.time() - t0)
    msg = f"[epoch {epoch_id+1}/demos {len(losses)}/time {time_elapse}] "
    if is_training:
        msg += "Training "
    else:
        msg += "Testing "
    msg += f"Loss (me,std,ma,mi): {avg_loss}, {std_loss}, {max_loss}, {min_loss}"

    log_msg(fp, msg)

# Prepare the vlm input for one demo (successful trajectory) such that
# * call the VLM on the entire token sequence once
# * use attention_mask to facilitate the retrival of each completed subgoal sample
def prepare_vlm_input_per_demo(demo, abstract_history, lowlevel_instr_set, tokenizer):
    obss = []

    # Walk around:
    #   The observation before the time step when a subgoal completes is used to verify the completion
    #   so, currently they are stored together in the collected successful tracjectoies.
    #   While, in VLM, the observation at the time step when the subgoal actually completes is used as
    #   a reference point. So, here, add 1 to 'time_step' to walk around this issue.
    time_step = 1
    pre_csg_time_step = 0
    critical_time_steps = 0 # count the time steps when some subgoal(s) completes
    mission = demo[0][0]['mission']
    input_text_seq = mission+"<image>"
    pre_last_time_step = len(demo[:-2])-1 # the last observation is not stored in the trajectory
    for (obs, _, _, completed_subgoals) in demo[:-2]:
        if not abstract_history:
            obss.append(obs)
        if len(completed_subgoals):
            critical_time_steps += 1
            focused_time_steps = time_step-pre_csg_time_step
            if abstract_history:
                focused_time_steps = 1
                # 'obs' represents the observation at 'time_step-1'
                # The observation when the subgoal completes is at 'time_step'
                # But the last observation is not needed
                if pre_csg_time_step!=pre_last_time_step:
                    obs = demo[time_step][0]
                obss.append(obs)
            input_text_seq += lowlevel_instr_set.get_completed_subgoals_msg(completed_subgoals)
            input_text_seq += "<"+"image"*focused_time_steps+">"
            pre_csg_time_step = time_step
        time_step += 1

    max_token_seq_len = 512
    vlm_input = tokenizer(input_text_seq, max_length=max_token_seq_len, padding="max_length", return_tensors='pt')
    input_token_seq_len = vlm_input['attention_mask'].sum(dim=-1)[0] # batch size is 1 here
    for key in vlm_input:
        vlm_input[key] = vlm_input[key].to(device)

    # Compute media_locations, labels, instance_weights
    media_locations = torch.zeros(vlm_input['input_ids'].shape, dtype=torch.bool, device=device)
    label_masks = torch.zeros(vlm_input['input_ids'].shape, dtype=torch.bool, device=device)
    instance_weights = torch.zeros(vlm_input['input_ids'].shape, dtype=torch.float, device=device)
    media_seq_start, media_seq_end = None, None
    # <image>: 27,  9060,    29
    tidx = 0
    while tidx < input_token_seq_len:
        if vlm_input['input_ids'][0, tidx] == 27: # '<'
            # media_seq_end==None corresponds to the indentification of the first image
            # That does not have any subgoal completed before it.
            if media_seq_end is not None:
                # store the labels of the passed text section
                label_masks[0, (media_seq_end+1):tidx] = True
                instance_weights[0, (media_seq_end+1):tidx] = 1.0/(tidx-media_seq_end-1)
            media_locations[0, tidx] = True
            media_seq_start = tidx

            # by pass the first 'image' token whose media location is being taken charge by
            # the prefix token, '<'
            tidx += 1
        elif vlm_input['input_ids'][0, tidx] == 9060: # 'image'
            media_locations[0, tidx] = True
        elif vlm_input['input_ids'][0, tidx] == 29: # '>'
            media_seq_end = tidx

        tidx += 1

    vlm_input['media_locations'] = media_locations
    vlm_input['instance_weights'] = instance_weights
    pad_label=-1
    vlm_input['labels'] = vlm_input['input_ids'].masked_fill(label_masks==0, pad_label)        

    return vlm_input, obss, critical_time_steps

def prepare_dataset(demos, abstract_history, lowlevel_instr_set, tokenizer):
    result = []
    for demo in demos:
        vlm_input, vlm_media, num_completed_subgoals = prepare_vlm_input_per_demo(
            demo, abstract_history, lowlevel_instr_set, tokenizer)
        result.append((vlm_input, vlm_media, num_completed_subgoals))
    return result

def train_test_helper(
    is_training,
    training_status_path,
    epoch_id,
    demos,
    log_interval,
    abstract_history,
    lowlevel_instr_set,
    tokenizer,
    vlm,
    bow_image_conv_encoder,
    optimizer=None,
    max_grad_norm=None):

    msg = ""
    t0 = time.time()
    losses=[]

    demo_ids = np.arange(0, len(demos))
    if is_training: # training
        demo_ids = np.random.permutation(demo_ids)
        vlm.train()
        bow_image_conv_encoder.train()
        msg = "Training..."
    else: # testing
        vlm.eval()
        bow_image_conv_encoder.eval()
        msg = "Testing..."
    
    num_log_intervals = len(demos)//log_interval
    log_msg(training_status_path, msg)

    for log_interval_idx in range(num_log_intervals):
        start_idx = log_interval_idx*args.log_interval
        end_idx = (1+log_interval_idx)*args.log_interval
        if log_interval_idx+1 != num_log_intervals:
            demos_interval = [demos[idx] for idx in demo_ids[start_idx:end_idx]]
        else:
            demos_interval = [demos[idx] for idx in demo_ids[start_idx:]]
        dataset = prepare_dataset(demos_interval, abstract_history, lowlevel_instr_set, tokenizer)
        
        num_demos = len(dataset)
        for demo_idx in range(num_demos):
            vlm_input, vlm_media, num_completed_subgoals = dataset[demo_idx]
            if is_training:
                # Calculate the training loss
                with amp.autocast(enabled=True):
                    vlm_input['image_embeds'] = bow_image_conv_encoder(vlm_media)
                    result = vlm(**vlm_input, return_dict=True)
                    loss = result['loss']/num_completed_subgoals
                # update the model(s)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(vlm.parameters(), max_grad_norm)
                torch.nn.utils.clip_grad_norm_(bow_image_conv_encoder.parameters(), max_grad_norm)
                optimizer.step()
            else:
                # Calculate the testing loss
                with torch.no_grad():
                    vlm_input['image_embeds'] = bow_image_conv_encoder(vlm_media)
                    result = vlm(**vlm_input, return_dict=True)
                    loss = result['loss']/num_completed_subgoals
            losses.append(loss.item())

        # get statistics for each log interval during training
        if is_training:
            log_losses_stat(training_status_path, losses, t0, epoch_id, is_training)

        # manage memory
        del vlm_media
        del vlm_input
        del result
        gc.collect()
        torch.cuda.empty_cache()
    
    # logging the testing loss
    if not is_training:
        log_losses_stat(training_status_path, losses, t0, epoch_id, is_training)

    loss_statistics = get_stat(losses, rd=4, return_type='np')
    return loss_statistics

# Parse arguments
#parser = ArgumentParser()
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

model_dir = os.path.join(utils.storage_dir(), "models", model_name_prefix)
os.makedirs(model_dir)

demos_path = utils.get_demos_path(args.demos_name, args.env, origin=None, valid=False)
demos = utils.load_demos(demos_path)
training_status_path = os.path.join(model_dir, "training_status.txt")
vlm_model_path = os.path.join(model_dir, "vlm.pt")
image_conv_model_path = os.path.join(model_dir, "image_conv.pt")
demos_train_set_path = os.path.join(model_dir, "trainset.pkl") 
demos_test_set_path = os.path.join(model_dir, "testset.pkl") 


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

    #
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

for epoch_i in range(0, args.epochs):
    msg = '======== Epoch {:} / {:} ========'.format(epoch_i + 1, args.epochs)
    log_msg(training_status_path, msg)

    # Training
    is_training = True
    tr_losses_stat = train_test_helper(
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