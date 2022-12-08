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
parser.add_argument("--epochs", type=int, default=4,
                    help="number of epochs for training (default: 4)")
parser.add_argument("--abstract-history", action="store_true", default=False,
                    help="Allows you to switch between the full history and the abstraction of the full history")


'''
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
parser.add_argument("--save-interval", type=int, default=50,
                    help="number of updates between two saves (default: 50, 0 means no saving)")


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
'''

args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

vlm = None
# Create the VLM with randomly initialized parameters
if args.vlm_arc == "Flamingo":
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

    #
    vit = None

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
        perceiver_num_time_embeds = args.max_history_window_vlm,#16, 8, 128
        only_attend_immediate_media=args.abstract_history # True: Abstracted history, False: Full history
    )
else:
    raise("Unsupported VLM")

env = gym.make(args.env)
image_preproc = RawImagePreprocessor()
visual_observation_bow_flat_dim=147
image_conv = nn.Sequential(*[
            ImageBOWEmbedding(visual_observation_bow_flat_dim, vlm.wte.embedding_dim),
            nn.Conv2d(
                in_channels=vlm.wte.embedding_dim, out_channels=vlm.wte.embedding_dim,
                kernel_size=(3, 3), stride=1, padding=1),
            nn.BatchNorm2d(vlm.wte.embedding_dim),
            nn.ReLU(),
            nn.Conv2d(in_channels=vlm.wte.embedding_dim, out_channels=vlm.wte.embedding_dim, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(vlm.wte.embedding_dim),
            nn.ReLU(),
        ])
image_conv.apply(initialize_parameters)

# Load the demonstrations and split it into training, validation and testing partitions
# "--env", "BabyAI-UnlockLocalR2Dist-v0",
# "--demos_name", "UnlockLocalR2Dist_BotDemosfrom babyai.levels.verifier import LowlevelInstrSet_100000",
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


def get_image_embedding(image_conv, image_preproc, obss, device):
    batch_size = 1
    images = image_preproc(obss, device=device)
    images = images.unsqueeze(dim=0)
    images = images.to(device)
    images = rearrange(images, 'b t h w c -> (b t) c h w')
    image_embeds = image_conv(images)
    # convert to the shape : (batch_size, times, height*width, feature_dim)
    image_embeds = rearrange(image_embeds, '(b t) d h w-> b t (h w) d', b=batch_size)
    return image_embeds

lowlevel_instr_set = LowlevelInstrSet()

# Train and evaluate
#   demo: ((obs, action, done, completed_subgoals), reward, seed)
#       obs: {'image':, 'direction':, 'mission':}
#       action: an integer
#       done: true or false
#       completed_subgoals: [list of completed subgoals' indices at timestep t]
#       reward: a real number between 0 and 1
#       seed: an integer
parameters = list(vlm.parameters()) + list(image_conv.parameters())
optimizer = AdamW(parameters, lr=args.lr) # default lr is 5e-5
vlm.to(device)
image_conv.to(device)
for epoch_i in range(0, args.epochs):
    print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, args.epochs))
    t0 = time.time()
    total_train_loss = 0
    vlm.train()
    tr_loss=[]

    randmized_demo_ids = np.arange(0, len(demos_train))
    randmized_demo_ids = np.random.permutation(randmized_demo_ids)
    e_idx = 0
    for demo_id in randmized_demo_ids:
        demo = demos_train[demo_id]
        time_step = 1
        obss = []
        csg_texts = []
        csg_time_steps = []
        pre_csg_time_step = 0
        for (obs, _, _, completed_subgoals) in demo[:-2]:
            if not args.abstract_history:
                obss.append(obs)
            if len(completed_subgoals):
                focused_time_steps = time_step-pre_csg_time_step
                if args.abstract_history:
                    focused_time_steps = 1
                    obss.append(obs)
                csg_text = "<image> "*focused_time_steps+lowlevel_instr_set.get_completed_subgoals_msg(completed_subgoals)
                csg_texts.append(csg_text)
                # Walk around:
                #   The observation before the time step when a subgoal completes is used to verify the completion
                #   so, currently they are stored together in the collected successful tracjectoies.
                #   While, in VLM, the observation at the time step when the subgoal actually completes is used as
                #   a reference point. So, here, add 1 to 'time_step' to walk around this issue.
                csg_time_steps.append(time_step)
                pre_csg_time_step = time_step
            time_step += 1
        
        pre_csg_time_steps = [0]
        pre_csg_time_steps.extend(csg_time_steps[:-1])

        csg_texts_tokens = tokenizer(csg_texts, padding=True, return_tensors="pt")
        for key in csg_texts_tokens:
            csg_texts_tokens[key] = csg_texts_tokens[key].to(device)
        csg_tokens_len = csg_texts_tokens['attention_mask'].sum(dim=-1)
        img_embeds = get_image_embedding(image_conv, image_preproc, obss, device)
        num_csgs = len(csg_time_steps)

        print(f"[{e_idx}][demo {demo_id}] {num_csgs}/{time_step - 1}")

        time_step = 0
        loss = None
        mission = demo[0][0]['mission']

        vlm_input = tokenizer([mission+"<image> "], max_length=512, padding="max_length", return_tensors='pt')
        for key in vlm_input:
            vlm_input[key] = vlm_input[key].to(device)
        input_text_len = vlm_input['attention_mask'].sum(dim=-1)[0] # batch size is 1 here
        vlm_input['media_locations'] = torch.zeros(vlm_input['input_ids'].shape, dtype=torch.bool, device=device)
        vlm_input['media_locations'][0, [input_text_len-4]] = True # '<image> ' (appended to mission) has four tokens
        vlm_input['image_embeds'] = img_embeds[:, [0], :, :]

        #pre_pre_csg_time_step = -1
        for idx, pre_csg_time_step, csg_time_step in zip(range(num_csgs), pre_csg_time_steps, csg_time_steps):
            # accumulate text tokens (target/completed subgoal) and image embedding
            # passed_steps: focused steps between two adjecent completed subgoals
            passed_steps = csg_time_step - pre_csg_time_step
            if args.abstract_history:
                passed_steps = 1
            subgoal_tokens_len = csg_tokens_len[idx]-3*passed_steps
            vlm_input['input_ids'][0, input_text_len:input_text_len+subgoal_tokens_len] = csg_texts_tokens['input_ids'][idx, (3*passed_steps):csg_tokens_len[idx]]
            vlm_input['attention_mask'][0, input_text_len:input_text_len+subgoal_tokens_len] = 1

            label_mask = torch.zeros(vlm_input['input_ids'].shape, dtype=torch.bool, device=device)
            label_mask[0, input_text_len:input_text_len+subgoal_tokens_len] = True

            pad_label=-1
            vlm_input['labels'] = vlm_input['input_ids'].masked_fill(label_mask==0, pad_label)

            #for key in vlm_input:
            #    vlm_input[key] = vlm_input[key].to(device)

            # Calculate the loss and update the model
            with amp.autocast(enabled=True):
                result = vlm(**vlm_input, return_dict=True)
                gc.collect()
                torch.cuda.empty_cache()

                loss_csg = result['loss']
                if loss:
                    loss += loss_csg
                else:
                    loss = loss_csg
            
            # append image embeddings for the next target subgoal if it exist
            input_text_len += subgoal_tokens_len # for the subgoal tokens
            #pre_pre_csg_time_step = pre_csg_time_step
            if idx+1 == num_csgs: # the last completed subgoal has been processed
                break

            if args.abstract_history:
                current_media_locations = [input_text_len]
                vlm_input['image_embeds'] = img_embeds[:, :(idx+2), :, :]
            else:
                current_media_locations = [input_text_len+3*i for i in range(0, passed_steps)]
                vlm_input['image_embeds'] = img_embeds[:, :(csg_time_step+1), :, :]

            vlm_input['input_ids'][0, input_text_len:input_text_len+(3*passed_steps)] = csg_texts_tokens['input_ids'][idx, :(3*passed_steps)]
            vlm_input['attention_mask'][0, input_text_len:input_text_len+(3*passed_steps)] = 1
            input_text_len += 3*passed_steps # for tokens of '<image> ...<image> ' preceding the subgoal

            vlm_input['media_locations'][0, current_media_locations] = True



        loss /= num_csgs
        tr_loss.append(loss.item())
        print(f"\t{loss.item()}")
        #print(f"[{e_idx}][demo {demo_id}] {loss.item()}")
        e_idx += 1

        # update the Flamingo model
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        gc.collect()
        torch.cuda.empty_cache()


    avg_train_loss = np.mean(tr_loss)
    std_train_loss = np.std(tr_loss)
    max_train_loss = np.max(tr_loss)
    min_train_loss = np.min(tr_loss) 
    training_time = format_time(time.time() - t0)
    gc.collect()
    print("[epoch {}] Training Loss (me,std,ma,mi): {0:.4f}, {0:.4f}, {0:.4f}, {0:.4f}".format(epoch_i + 1, avg_train_loss, std_train_loss, max_train_loss, min_train_loss))
    print("Training epoch took: {:}".format(training_time))

    '''
    t0 = time.time()
    vlm.eval()
    val_loss = 0
    losses = []
    for batch in val_loader:
        losses = []
        with torch.no_grad():
            batch_size = batch['input_ids'].size(0)
            for key in batch:
                batch[key] = batch[key].to(device)
            outputs= model(**batch, return_dict=True)          
            loss = outputs['loss']
            losses.append(loss.item())
    val_loss = np.mean(losses)
    gc.collect() 
    validation_time = format_time(time.time() - t0)
    print("Validation Loss: {0:.4f}".format(val_loss))
    print("Validation took: {:}".format(validation_time))
    print("")
    '''
