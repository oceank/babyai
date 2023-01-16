import os
import time
import datetime
import gc
import copy
from einops import rearrange
import torch
from torch.utils.data import Dataset
import torch.nn as nn
from torch.cuda import amp
import numpy as np
import matplotlib.pyplot as plt

from ..model import ImageBOWEmbedding, initialize_parameters


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
    
    # 'images' is a list of lists of images when self.image_preproc is not None else it is torch tensor with a shape, (batch, num_images_per_sample, ...)
    def forward(self, images):
        batch_size = len(images)
        if self.image_preproc:
            list_of_images_per_epsiode = [self.image_preproc(images_per_episode, device=self.device) for images_per_episode in images]
            image_tensors = torch.stack(list_of_images_per_epsiode, dim=0)
        else:
            image_tensors = images
        image_tensors = rearrange(image_tensors, 'b t h w c -> (b t) c h w')
        image_embeds = self.image_conv(image_tensors)
        # convert to the shape : (batch_size, times, height*width, feature_dim)
        image_embeds = rearrange(image_embeds, '(b t) d h w-> b t (h w) d', b=batch_size)
        return image_embeds

class SubgoalsDemoDataset(Dataset):
    def __init__(self, demos):
        self.demos = demos

    def __getitem__(self, idx):
        x = self.demos[idx]
        return x

    def __len__(self):
        return len(self.demos)

def subgoal_demo_collate_fn(demos, abstract_history, lowlevel_instr_set, tokenizer, image_preproc, skip_label, pin_memory):

    seeds = []
    all_obss = []
    all_pre_csg_time_steps = []
    input_text_seqs = []

    # Parse all episodes in the batch
    for demo in demos:
        obss, csg_texts, csg_time_steps, pre_csg_time_steps = prepare_input_seq(demo, abstract_history, lowlevel_instr_set)
        mission = demo[0][0]['mission']
        input_text_seq = f"Goal: {mission}" + "".join(csg_texts)

        all_obss.append(obss)
        all_pre_csg_time_steps.append(pre_csg_time_steps)
        input_text_seqs.append(input_text_seq)
        seeds.append(demo[-1])

    vlm_input = tokenizer(input_text_seqs, padding="longest", return_tensors='pt')

    media_locations, label_masks, instance_weights = cal_media_loc_labels_token_weights(vlm_input)
    vlm_input['media_locations'] = media_locations
    vlm_input['instance_weights'] = instance_weights
    vlm_input['labels'] = vlm_input['input_ids'].masked_fill(label_masks==0, skip_label)

    vlm_media = add_image_padding_and_preprocess(all_obss, image_preproc)

    if pin_memory:
        for key in vlm_input:
            vlm_input[key] = vlm_input[key].pin_memory()
        vlm_media = vlm_media.pin_memory()

    return vlm_input, vlm_media, seeds, all_pre_csg_time_steps


class SubgoalsDemoParsedDataset(Dataset):
    def __init__(self, demos, abstract_history, lowlevel_instr_set):
        self.seeds = []
        self.all_obss = []
        self.all_pre_csg_time_steps = []
        self.input_text_seqs = []

        # Parse all episodes in the batch
        for demo in demos:
            obss, csg_texts, csg_time_steps, pre_csg_time_steps = prepare_input_seq(demo, abstract_history, lowlevel_instr_set)
            mission = demo[0][0]['mission']
            input_text_seq = f"Goal: {mission}" + "".join(csg_texts)

            self.all_obss.append(obss)
            self.all_pre_csg_time_steps.append(pre_csg_time_steps)
            self.input_text_seqs.append(input_text_seq)
            self.seeds.append(demo[-1])

    def __getitem__(self, idx):
        seed = self.seeds[idx]
        input_text_seq = self.input_text_seqs[idx]
        pre_csg_time_steps = self.all_pre_csg_time_steps[idx]
        obss = self.all_obss[idx]
        return seed, input_text_seq, pre_csg_time_steps, obss

    def __len__(self):
        return len(self.seeds)

def subgoal_demo_parsed_collate_fn(batch, tokenizer, image_preproc, skip_label, pin_memory):

    seeds = []
    all_obss = []
    all_pre_csg_time_steps = []
    input_text_seqs = []

    for sample in batch:
        seeds.append(sample[0])
        input_text_seqs.append(sample[1])
        all_pre_csg_time_steps.append(sample[2])
        all_obss.append(sample[3])

    vlm_input = tokenizer(input_text_seqs, padding="longest", return_tensors='pt')

    media_locations, label_masks, instance_weights = cal_media_loc_labels_token_weights(vlm_input)
    vlm_input['media_locations'] = media_locations
    vlm_input['instance_weights'] = instance_weights
    vlm_input['labels'] = vlm_input['input_ids'].masked_fill(label_masks==0, skip_label)

    vlm_media = add_image_padding_and_preprocess(all_obss, image_preproc)

    if pin_memory:
        for key in vlm_input:
            vlm_input[key] = vlm_input[key].pin_memory()
        vlm_media = vlm_media.pin_memory()

    return vlm_input, vlm_media, seeds, all_pre_csg_time_steps

class SubgoalsDemoTokenizedDataset(Dataset):
    def __init__(self, demos, abstract_history, lowlevel_instr_set, tokenizer, skip_label):
        self.seeds = []
        self.all_obss = []
        self.all_pre_csg_time_steps = []
        self.input_text_seqs = []
        self.all_vlm_inputs = []
        self.input_seq_lens = []
        self.skip_label = skip_label

        # Parse all episodes in the batch
        for demo in demos:
            obss, csg_texts, csg_time_steps, pre_csg_time_steps = prepare_input_seq(demo, abstract_history, lowlevel_instr_set)
            mission = demo[0][0]['mission']
            input_text_seq = f"Goal: {mission}" + "".join(csg_texts)

            self.all_obss.append(obss)
            self.all_pre_csg_time_steps.append(pre_csg_time_steps)
            self.input_text_seqs.append(input_text_seq)
            self.seeds.append(demo[-1])

        for input_text_seq in self.input_text_seqs:
            vlm_input = tokenizer(input_text_seq, return_tensors='pt')
            self.input_seq_lens.append(vlm_input['input_ids'].shape[-1]) # the length of the input sequence
            self.all_vlm_inputs.append(vlm_input)

        for vlm_input in self.all_vlm_inputs:
            media_locations, label_masks, instance_weights = cal_media_loc_labels_token_weights(vlm_input)
            vlm_input['media_locations'] = media_locations
            vlm_input['instance_weights'] = instance_weights
            vlm_input['labels'] = vlm_input['input_ids'].masked_fill(label_masks==0, self.skip_label)

    def __getitem__(self, idx):
        seed = self.seeds[idx]
        obss = self.all_obss[idx]
        pre_csg_time_steps = self.all_pre_csg_time_steps[idx]
        input_text_seq = self.input_text_seqs[idx]
        vlm_input = self.all_vlm_inputs[idx]
        input_seq_len = self.input_seq_lens[idx]
        return seed, input_text_seq, pre_csg_time_steps, obss, vlm_input, input_seq_len

    def __len__(self):
        return len(self.seeds)

def subgoal_demo_tokenized_collate_fn(batch, tokenizer, image_preproc, skip_label, pin_memory):
    all_obss = []
    seeds = []
    all_pre_csg_time_steps = []
    max_seq_len = max([sample[5] for sample in batch])
    num_demos_in_batch = len(batch)

    vlm_input ={}
    vlm_input['input_ids'] = torch.full((num_demos_in_batch, max_seq_len), tokenizer.pad_token_id, dtype=torch.long)
    vlm_input['attention_mask'] = torch.full((num_demos_in_batch, max_seq_len), 0, dtype=torch.bool)
    vlm_input['media_locations'] = torch.full((num_demos_in_batch, max_seq_len), 0, dtype=torch.bool)
    vlm_input['labels'] = torch.full((num_demos_in_batch, max_seq_len), skip_label, dtype=torch.long)
    vlm_input['instance_weights'] = torch.full((num_demos_in_batch, max_seq_len), 0.0, dtype=torch.float)

    for idx, (seed, input_text_seq, pre_csg_time_steps, obss, vlm_input_episode, input_seq_len) in enumerate(batch):
        vlm_input['input_ids'][idx, :input_seq_len] = vlm_input_episode['input_ids'][0]
        vlm_input['attention_mask'][idx, :input_seq_len] = 1
        vlm_input['media_locations'][idx, :input_seq_len] = vlm_input_episode['media_locations'][0]
        vlm_input['labels'][idx, :input_seq_len] = vlm_input_episode['labels'][0]
        vlm_input['instance_weights'][idx, :input_seq_len] = vlm_input_episode['instance_weights'][0]
        all_obss.append(obss)
        seeds.append(seed)
        all_pre_csg_time_steps.append(pre_csg_time_steps)

    vlm_media = add_image_padding_and_preprocess(all_obss, image_preproc)

    if pin_memory:
        for key in vlm_input:
            vlm_input[key] = vlm_input[key].pin_memory()
        vlm_media = vlm_media.pin_memory()

    return vlm_input, vlm_media, seeds, all_pre_csg_time_steps

def format_time(elapsed):
    elapsed_rounded = int(round((elapsed)))
    return str(datetime.timedelta(seconds=elapsed_rounded))

def log_msg(fp, msg, logging_mode='a'):
    with open(fp, logging_mode) as f:
        f.write(msg + "\n")

# Input:
#   arr: a list of numbers
def get_stat(arr, rd=6, return_type='plain'):
    assert isinstance(arr, list)

    ave = round(np.mean(arr),rd)
    std = round(np.std(arr), rd)
    max = round(np.max(arr), rd)
    min = round(np.min(arr), rd)
    if return_type == "np":
        return np.array([ave, std, max, min])
    else:
        return ave, std, max, min

def log_losses_stat(fp, losses, t0, epoch_id, is_training, rd=6):
    avg_loss, std_loss, max_loss, min_loss = get_stat(losses, rd=rd)
    time_elapse = format_time(time.time() - t0)
    msg = f"[epoch {epoch_id+1}/demos {len(losses)}/time {time_elapse}] "
    if is_training:
        msg += "Training "
    else:
        msg += "Testing "
    msg += f"Loss (me,std,ma,mi): {avg_loss}, {std_loss}, {max_loss}, {min_loss}"

    log_msg(fp, msg)

# Output:
#   obss: list of (critical) observations.
#       The last observation of a successful trajectory is not saved.
#       The information of a completed subgoal (at time step t) is saved together with the observation at time step (t-1)
#   csg_time_steps: list of time steps when a subgoal completes.
#   csg_texts: list of 'csg_text'
#       csg_text: "completed_subgoal"|image...image|.
#       The # of 'image' indicates the # of elapsed time steps since the last completed subgoal/mission starts.
# Note:
# The format of input sequence to the VLM:
#   Goal: 'mission'|image|[None]'sg1'|image...image|[Success]'sg2'...
def prepare_input_seq(demo, abstract_history, lowlevel_instr_set):
    # Walk around:
    #   The observation before the time step when a subgoal completes is used to verify the completion
    #   so, currently they are stored together in the collected successful tracjectoies.
    #   While, in VLM, the observation at the time step when the subgoal actually completes is used as
    #   a reference point. So, here, add 1 to 'time_step' to walk around this issue.
    time_step = 1
    pre_csg_time_step = 0
    pre_pre_csg_time_step = -1
    obss = []
    csg_texts = []
    csg_time_steps = []
    pre_csg_time_steps = []
    total_time_steps = len(demo[:-2]) # 0 based

    if not abstract_history:
        obss = [obs for (obs, _, _, _) in demo[:-2]]

    num_csg = 0
    for t in range(0, total_time_steps):
        completed_subgoals = demo[t][3]

        if len(completed_subgoals):
            # number of elapsed visual observations for completing the previous subgoal 
            num_attended_vis_obss = pre_csg_time_step - pre_pre_csg_time_step
            if abstract_history:
                obs = demo[pre_csg_time_step][0]
                obss.append(obs)
                num_attended_vis_obss = 1

            csg_text = "|"+"image"*num_attended_vis_obss+"|"
            if num_csg != 0:
                csg_text += f"[Success]"
            else:
                csg_text += "[None]"
            csg_text += lowlevel_instr_set.get_completed_subgoals_msg(completed_subgoals)
            csg_texts.append(csg_text)
            csg_time_steps.append(time_step)
            pre_csg_time_steps.append(pre_csg_time_step)
            pre_pre_csg_time_step = pre_csg_time_step
            pre_csg_time_step = time_step
            num_csg += 1

        time_step += 1

    return  obss, csg_texts, csg_time_steps, pre_csg_time_steps

# === Training by separately processing each episode ===
def prepare_vlm_input_per_demo(device, demo, abstract_history, lowlevel_instr_set, tokenizer, skip_label=-1, max_token_seq_len = 512):
    # Prepare the vlm input for one demo (successful trajectory) such that
    # * call the VLM on the entire token sequence once
    # * use attention_mask to facilitate the retrival of each completed subgoal sample

    seed = demo[-1]
    obss = []

    obss, csg_texts, csg_time_steps, pre_csg_time_steps = prepare_input_seq(demo, abstract_history, lowlevel_instr_set)
    mission = demo[0][0]['mission']
    input_text_seq = f"Goal: {mission}" + "".join(csg_texts)

    if max_token_seq_len <= 0:
        vlm_input = tokenizer(input_text_seq, padding="longest", return_tensors='pt')
    else:
        vlm_input = tokenizer(input_text_seq, max_length=max_token_seq_len, padding="max_length", return_tensors='pt')
    input_token_seq_len = vlm_input['attention_mask'].sum(dim=-1)[0] # batch size is 1 here
    for key in vlm_input:
        vlm_input[key] = vlm_input[key].to(device)

    media_locations, label_masks, instance_weights = cal_media_loc_labels_token_weights(vlm_input, device)
    vlm_input['media_locations'] = media_locations
    vlm_input['instance_weights'] = instance_weights
    vlm_input['labels'] = vlm_input['input_ids'].masked_fill(label_masks==0, skip_label)
    return vlm_input, obss, seed, pre_csg_time_steps

def prepare_dataset(device, demos, abstract_history, lowlevel_instr_set, tokenizer, skip_label=-1):
    result = []
    for demo in demos:
        vlm_input, vlm_media, seed, pre_csg_time_steps = prepare_vlm_input_per_demo(
            device, demo, abstract_history, lowlevel_instr_set, tokenizer, skip_label, max_token_seq_len=0)
        result.append((vlm_input, vlm_media, seed, pre_csg_time_steps))
    return result

def train_test_helper(
    device,
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
    skip_label=-1,
    optimizer=None,
    max_grad_norm=None,
    batch_size=1):

    vlm.to(device)
    bow_image_conv_encoder.to(device)

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

    num_batches = len(demos)//batch_size
    if len(demos)%batch_size != 0:
        num_batches += 1

    log_msg(training_status_path, msg)

    fc_time_elapse = 0.0
    bo_time_elapse = 0.0

    processed_demos = 0
    for batch_idx in range(num_batches):
        start_idx = batch_idx*batch_size
        end_idx = (1+batch_idx)*batch_size
        if batch_idx+1 != num_batches:
            demos_to_process = [demos[idx] for idx in demo_ids[start_idx:end_idx]]
        else:
            demos_to_process = [demos[idx] for idx in demo_ids[start_idx:]]
        dataset = prepare_dataset(
            device, demos_to_process, abstract_history,
            lowlevel_instr_set, tokenizer, skip_label)

        batch_loss = 0 # only used for training
        num_demos = len(dataset)
        if is_training:
            optimizer.zero_grad()
        tt = time.time()
        for demo_idx in range(num_demos):
            vlm_input, vlm_media, seed, pre_csg_time_steps = dataset[demo_idx]
            # Calculate the loss
            with amp.autocast(enabled=True) if is_training else torch.no_grad():
                vlm_input['image_embeds'] = bow_image_conv_encoder([vlm_media])
                result = vlm(**vlm_input, return_dict=True)
                loss = result['loss']
                batch_loss += loss
            losses.append(loss.item())
        fc_time_elapse += time.time()-tt

        if is_training:
            tt = time.time()
            # update the model(s) after processing a batch of episodes
            batch_loss /= num_demos
            batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(vlm.parameters(), max_grad_norm)
            torch.nn.utils.clip_grad_norm_(bow_image_conv_encoder.parameters(), max_grad_norm)
            optimizer.step()
            bo_time_elapse += time.time() - tt

        processed_demos += num_demos

        # get statistics for each log interval during training
        if is_training and (processed_demos%log_interval==0 or processed_demos==len(demos)):
            log_losses_stat(training_status_path, losses, t0, epoch_id, is_training, rd=6)

    print(f"Forward Call: {fc_time_elapse}")
    print(f"Backward call and Optimization: {bo_time_elapse}")
    # logging the testing loss
    if not is_training:
        log_losses_stat(training_status_path, losses, t0, epoch_id, is_training, rd=6)
    
    gc.collect()
    torch.cuda.empty_cache()

    loss_statistics = get_stat(losses, rd=6, return_type='np')
    return loss_statistics

# === Training by separately processing a batch of episodes ===
def parse_collected_demos_per_demo(device, demos, abstract_history, lowlevel_instr_set, tokenizer, skip_label=-1):

    seeds = []
    all_obss = []
    all_pre_csg_time_steps = []
    input_text_seqs = []
    all_vlm_inputs = []
    input_seq_lens = []

    t0 = time.time()
    # Parse all episodes in the batch
    for demo in demos:
        obss, csg_texts, csg_time_steps, pre_csg_time_steps = prepare_input_seq(demo, abstract_history, lowlevel_instr_set)
        mission = demo[0][0]['mission']
        input_text_seq = f"Goal: {mission}" + "".join(csg_texts)

        all_obss.append(obss)
        all_pre_csg_time_steps.append(pre_csg_time_steps)
        input_text_seqs.append(input_text_seq)
        seeds.append(demo[-1])
    time_elapse = format_time(time.time()-t0)
    print(f"Episodes Parsing: {time_elapse}")

    t0 = time.time()
    for input_text_seq in input_text_seqs:
        vlm_input = tokenizer(input_text_seq, return_tensors='pt')
        input_seq_lens.append(vlm_input['input_ids'].shape[-1]) # the length of the input sequence
        all_vlm_inputs.append(vlm_input)
    time_elapse = format_time(time.time()-t0)
    print(f"Tokenization: {time_elapse}")

    t0 = time.time()
    for vlm_input in all_vlm_inputs:
        for key in vlm_input:
            vlm_input[key] = vlm_input[key].to(device)
    time_elapse = format_time(time.time()-t0)
    print(f"To Device {device}: {time_elapse}")

    t0 = time.time()
    for vlm_input in all_vlm_inputs:
        media_locations, label_masks, instance_weights = cal_media_loc_labels_token_weights(vlm_input, device)
        vlm_input['media_locations'] = media_locations
        vlm_input['instance_weights'] = instance_weights
        vlm_input['labels'] = vlm_input['input_ids'].masked_fill(label_masks==0, skip_label)
    time_elapse = format_time(time.time()-t0)
    print(f"Compute labels, media_locations and instance_weights: {time_elapse}")

    batch_sizes = [2, 5, 10, 20, 40]
    num_all_demos = len(seeds)
    for batch_size in batch_sizes:
        t0 = time.time()
        demo_ids = np.arange(num_all_demos)
        demo_ids = np.random.permutation(demo_ids)
        num_batches = num_all_demos//batch_size
        if len(demos)%batch_size != 0:
            num_batches += 1
        for batch_idx in range(num_batches):
            start_idx = batch_idx*batch_size
            end_idx = (1+batch_idx)*batch_size
            if batch_idx+1 != num_batches:
                demo_ids_batch = demo_ids[start_idx:end_idx]
            else:
                demo_ids_batch = demo_ids[start_idx:]
            max_seq_len = max([input_seq_lens[i] for i in demo_ids_batch])
            num_demos_in_batch = len(demo_ids_batch)
            vlm_input ={}
            vlm_input['input_ids'] = torch.full((num_demos_in_batch, max_seq_len), tokenizer.pad_token_id, dtype=torch.long, device=device)
            vlm_input['attention_mask'] = torch.full((num_demos_in_batch, max_seq_len), 0, dtype=torch.bool, device=device)
            vlm_input['media_locations'] = torch.full((num_demos_in_batch, max_seq_len), 0, dtype=torch.bool, device=device)
            vlm_input['labels'] = torch.full((num_demos_in_batch, max_seq_len), skip_label, dtype=torch.long, device=device)
            vlm_input['instance_weights'] = torch.full((num_demos_in_batch, max_seq_len), 0.0, dtype=torch.float, device=device)

            for idx, demo_id in enumerate(demo_ids_batch):
                vlm_input['input_ids'][idx, :input_seq_lens[demo_id]] = all_vlm_inputs[demo_id]['input_ids'][0]
                vlm_input['attention_mask'][idx, :input_seq_lens[demo_id]] = 1
                vlm_input['media_locations'][idx, :input_seq_lens[demo_id]] = all_vlm_inputs[demo_id]['media_locations'][0]
                vlm_input['labels'][idx, :input_seq_lens[demo_id]] = all_vlm_inputs[demo_id]['labels'][0]
                vlm_input['instance_weights'][idx, :input_seq_lens[demo_id]] = all_vlm_inputs[demo_id]['instance_weights'][0]
        time_elapse = format_time(time.time()-t0)
        print(f"Pad labels, media_locations and instance_weights[Batch{batch_size}]: {time_elapse}")

    return all_vlm_inputs, all_obss, seeds, all_pre_csg_time_steps

def parse_collected_demos(device, demos, abstract_history, lowlevel_instr_set, tokenizer, skip_label=-1, max_token_seq_len = 512):

    seeds = []
    all_obss = []
    all_pre_csg_time_steps = []
    input_text_seqs = []

    t0 = time.time()
    # Parse all episodes in the batch
    for demo in demos:
        obss, csg_texts, csg_time_steps, pre_csg_time_steps = prepare_input_seq(demo, abstract_history, lowlevel_instr_set)
        mission = demo[0][0]['mission']
        input_text_seq = f"Goal: {mission}" + "".join(csg_texts)

        all_obss.append(obss)
        all_pre_csg_time_steps.append(pre_csg_time_steps)
        input_text_seqs.append(input_text_seq)
        seeds.append(demo[-1])
    time_elapse = format_time(time.time()-t0)
    print(f"Episodes Parsing: {time_elapse}")

    t0 = time.time()
    vlm_input = tokenizer(input_text_seqs, max_length=max_token_seq_len, padding="max_length", return_tensors='pt')
    time_elapse = format_time(time.time()-t0)
    print(f"Tokenization: {time_elapse}")

    t0 = time.time()
    for key in vlm_input:
        vlm_input[key] = vlm_input[key].to(device)
    time_elapse = format_time(time.time()-t0)
    print(f"To Device {device}: {time_elapse}")

    t0 = time.time()
    media_locations, label_masks, instance_weights = cal_media_loc_labels_token_weights(vlm_input, device)
    vlm_input['media_locations'] = media_locations
    vlm_input['instance_weights'] = instance_weights
    vlm_input['labels'] = vlm_input['input_ids'].masked_fill(label_masks==0, skip_label)
    time_elapse = format_time(time.time()-t0)
    print(f"Compute labels, media_locations and instance_weights: {time_elapse}")

    '''
    all_obss = add_image_padding(all_obss)
    '''
    return vlm_input, all_obss, seeds, all_pre_csg_time_steps

def add_image_padding(all_obss):
    result = []
    # conduct image padding for each episode when necessary
    max_num_images = max([len(obss) for obss in all_obss])
    for obss in all_obss:
        padded_obss = copy.copy(obss)
        if len(obss) < max_num_images:
            obs_padding = {
                'image': np.zeros(obss[0]['image'].shape, dtype=np.uint8),
                'direction':0,
                'mission':obss[0]['mission']
            }
            padded_obss.extend([obs_padding]*(max_num_images-len(obss)))
        result.append(padded_obss)
    return result

def add_image_padding_and_preprocess(all_obss, image_preproc):
    # all_obss: a list of lists of observations
    image_tensors = None
    # conduct image padding for each episode when necessary
    max_num_images = max([len(obss) for obss in all_obss])
    batch_size = len(all_obss)
    image_shape = all_obss[0][0]['image'].shape # check the first observation of the first episode
    new_shape = [batch_size, max_num_images]
    new_shape.extend(image_shape)
    image_tensors = torch.zeros(new_shape, dtype=torch.uint8)
    for bidx, obss in enumerate(all_obss):
        obss_tensor = image_preproc(obss)
        image_tensors[bidx, :len(obss)] = obss_tensor[0]

    return image_tensors

def cal_media_loc_labels_token_weights(vlm_input, device=torch.device("cpu")):
    batch_size = vlm_input['input_ids'].shape[0]
    input_token_seq_lens = vlm_input['attention_mask'].sum(dim=-1)

    # Compute media_locations, labels, instance_weights
    media_locations = torch.zeros(vlm_input['input_ids'].shape, dtype=torch.bool, device=device)
    label_masks = torch.zeros(vlm_input['input_ids'].shape, dtype=torch.bool, device=device)
    instance_weights = torch.zeros(vlm_input['input_ids'].shape, dtype=torch.float, device=device)
    sym_bar_locations = (vlm_input['input_ids'] == 91) # '|'

    for b_idx in range(batch_size):
        input_ids_len = input_token_seq_lens[b_idx]
        sym_bar_inds_1 = []
        for t_idx in range(input_ids_len):
            if sym_bar_locations[b_idx, t_idx]:
                sym_bar_inds_1.append(t_idx)
        sym_bar_inds_2 = sym_bar_inds_1[1:]
        sym_bar_inds_2.append(input_ids_len) # input_ids_len indicates the index of a virtual '|' behind the last non-masked token
        for i in range(len(sym_bar_inds_1)):
            if i%2 == 0:
                media_locations[b_idx, sym_bar_inds_1[i]] = True
                media_locations[b_idx, sym_bar_inds_1[i]+2:sym_bar_inds_2[i]] = True
            else:
                label_masks[b_idx, (sym_bar_inds_1[i]+4):sym_bar_inds_2[i]] = True
                instance_weights[b_idx, (sym_bar_inds_1[i]+4):sym_bar_inds_2[i]] = 1.0/(sym_bar_inds_2[i]-sym_bar_inds_1[i]-4)
    return media_locations, label_masks, instance_weights

def cal_media_loc_labels_token_weights_slow(vlm_input, device=torch.device("cpu")):
    # deprecated function for reference purpose only

    batch_size = vlm_input['input_ids'].shape[0]
    input_token_seq_lens = vlm_input['attention_mask'].sum(dim=-1)

    # Compute media_locations, labels, instance_weights
    media_locations = torch.zeros(vlm_input['input_ids'].shape, dtype=torch.bool, device=device)
    label_masks = torch.zeros(vlm_input['input_ids'].shape, dtype=torch.bool, device=device)
    instance_weights = torch.zeros(vlm_input['input_ids'].shape, dtype=torch.float, device=device)

    for b_idx in range(batch_size):
        media_seq_open = False
        media_seq_end = None
        # |image|: 91,  9060,  91
        # Subgoal Status: '[Success]', '[Failure]'
        num_tokens_sugboal_status = 3
        t_idx = 0
        while t_idx < input_token_seq_lens[b_idx]:
            if vlm_input['input_ids'][b_idx, t_idx] == 91: # token 91 indicatas a '|'
                if not media_seq_open:
                    # media_seq_end==None corresponds to the indentification of the first image
                    # That does not have any subgoal completed before it.
                    if media_seq_end is not None:
                        # store the labels of the passed text section
                        label_start_indice = media_seq_end+1+num_tokens_sugboal_status
                        label_masks[b_idx, label_start_indice:t_idx] = True
                        instance_weights[b_idx, label_start_indice:t_idx] = 1.0/(t_idx-label_start_indice)
                    media_locations[b_idx, t_idx] = True

                    # by pass the first 'image' token whose media location is being taken charge by
                    # the prefix token, '|'
                    t_idx += 1
                    media_seq_open = True
                else:
                    media_seq_end = t_idx
                    media_seq_open = False
            elif vlm_input['input_ids'][b_idx, t_idx] == 9060: # 'image'
                media_locations[b_idx, t_idx] = True

            t_idx += 1

        # save the labels of the last subgoal
        label_start_indice = media_seq_end+1+num_tokens_sugboal_status
        label_masks[b_idx, label_start_indice:t_idx] = True
        instance_weights[b_idx, label_start_indice:t_idx] = 1.0/(t_idx-label_start_indice)
    return media_locations, label_masks, instance_weights


def prepare_vlm_input_per_batch(device, demos, abstract_history, lowlevel_instr_set, tokenizer, skip_label=-1, max_token_seq_len = 512):
    # Prepare the vlm input for one batch of expisodes

    seeds = []
    all_obss = []
    all_pre_csg_time_steps = []
    input_text_seqs = []

    # Parse all episodes in the bach
    for demo in demos:
        obss, csg_texts, csg_time_steps, pre_csg_time_steps = prepare_input_seq(demo, abstract_history, lowlevel_instr_set)
        mission = demo[0][0]['mission']
        input_text_seq = f"Goal: {mission}" + "".join(csg_texts)

        all_obss.append(obss)
        all_pre_csg_time_steps.append(pre_csg_time_steps)
        input_text_seqs.append(input_text_seq)
        seeds.append(demo[-1])

    vlm_input = tokenizer(input_text_seqs, max_length=max_token_seq_len, padding="max_length", return_tensors='pt')
    for key in vlm_input:
        vlm_input[key] = vlm_input[key].to(device)

    # Calculate media locations, labels and token weights
    media_locations, label_masks, instance_weights = cal_media_loc_labels_token_weights(vlm_input, device)
    vlm_input['media_locations'] = media_locations
    vlm_input['instance_weights'] = instance_weights
    vlm_input['labels'] = vlm_input['input_ids'].masked_fill(label_masks==0, skip_label)

    # conduct image padding for each episode when necessary
    all_obss = add_image_padding(all_obss)

    return vlm_input, all_obss, seeds, all_pre_csg_time_steps

def train_test_helper_batch_process(
    device,
    is_training,
    training_status_path,
    epoch_id,
    dataset,
    log_interval,
    vlm,
    bow_image_conv_encoder,
    optimizer=None,
    max_grad_norm=None,
    batch_size=1):

    vlm.to(device)
    bow_image_conv_encoder.to(device)

    msg = ""
    t0 = time.time()
    losses=None

    if is_training: # training
        demo_ids = np.random.permutation(demo_ids)
        vlm.train()
        bow_image_conv_encoder.train()
        msg = "Training..."
    else: # testing
        vlm.eval()
        bow_image_conv_encoder.eval()
        msg = "Testing..."

    all_vlm_input, all_vlm_media, seeds, all_pre_csg_time_steps = dataset
    num_all_demos = len(seeds)
    demo_ids = np.arange(0, num_all_demos)
    num_batches = num_all_demos//batch_size
    if num_all_demos%batch_size != 0:
        num_batches += 1

    log_msg(training_status_path, msg)

    td_time_elapse = 0.0
    ip_time_elapse = 0.0
    fc_time_elapse = 0.0
    bo_time_elapse = 0.0

    processed_demos = 0
    for batch_idx in range(num_batches):
        start_idx = batch_idx*batch_size
        end_idx = (1+batch_idx)*batch_size
        if batch_idx+1 != num_batches:
            demo_ids_batch = demo_ids[start_idx:end_idx]
        else:
            demo_ids_batch = demo_ids[start_idx:]

        num_demos = len(demo_ids_batch)
        vlm_input = {}
        tt = time.time()
        for key in all_vlm_input:
            vlm_input[key] = all_vlm_input[key][demo_ids_batch]
            vlm_input[key] = vlm_input[key].to(device) # move to GPU
        td_time_elapse += time.time()-tt
        #print(f"To device {device}: {time_elapse}")
        tt = time.time()
        vlm_media = [all_vlm_media[i] for i in demo_ids_batch]
        vlm_media = add_image_padding(vlm_media)
        ip_time_elapse += time.time()-tt

        # Calculate the batch loss
        if is_training:
            optimizer.zero_grad()
        tt = time.time()
        with amp.autocast(enabled=True) if is_training else torch.no_grad():
            vlm_input['image_embeds'] = bow_image_conv_encoder(vlm_media)
            result = vlm(**vlm_input, return_dict=True)
            batch_loss = result['loss']
        fc_time_elapse += time.time()-tt
        #print(f"Forward Call: {time_elapse}")

        if losses is not None:
            losses.extend(batch_loss.detach().clone().tolist())
        else:
            losses = batch_loss.detach().clone().tolist()

        if is_training:
            tt = time.time()
            # update the model(s) after processing a batch of episodes
            loss = batch_loss.mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(vlm.parameters(), max_grad_norm)
            torch.nn.utils.clip_grad_norm_(bow_image_conv_encoder.parameters(), max_grad_norm)
            optimizer.step()
            bo_time_elapse += time.time()-tt
            #print(f"Backward call and Optimization: {time_elapse}")

        processed_demos += num_demos

        # get statistics for each log interval during training
        if is_training and (processed_demos%log_interval==0 or processed_demos==num_all_demos):
            log_losses_stat(training_status_path, losses, t0, epoch_id, is_training, rd=6)
    
    print(f"To device {device}: {format_time(td_time_elapse)}")
    print(f"Image Padding: {format_time(ip_time_elapse)}")
    print(f"Forward Call: {format_time(fc_time_elapse)}")
    print(f"Backward call and Optimization: {format_time(bo_time_elapse)}")

    # logging the testing loss
    if not is_training:
        log_losses_stat(training_status_path, losses, t0, epoch_id, is_training, rd=6)
    
    gc.collect()
    torch.cuda.empty_cache()

    loss_statistics = get_stat(losses, rd=6, return_type='np')
    return loss_statistics


def train_test_helper_batch_process_with_dataloader(
    device,
    is_training,
    training_status_path,
    epoch_id,
    dataloader,
    num_all_demos,
    log_interval,
    vlm,
    bow_image_conv_encoder,
    optimizer=None,
    max_grad_norm=None):

    vlm.to(device)
    bow_image_conv_encoder.to(device)

    msg = ""
    t0 = time.time()
    losses=None

    if is_training: # training
        vlm.train()
        bow_image_conv_encoder.train()
        msg = "Training..."
    else: # testing
        vlm.eval()
        bow_image_conv_encoder.eval()
        msg = "Testing..."

    log_msg(training_status_path, msg)

    processed_demos = 0
    for batch in dataloader:
        vlm_input, vlm_media, seeds, all_pre_csg_time_steps = batch
        num_demos = vlm_media.shape[0]
        # move data to GPU
        for key in vlm_input:
            vlm_input[key] = vlm_input[key].to(device)
        vlm_media = vlm_media.to(device)

        # Calculate the batch loss
        if is_training:
            optimizer.zero_grad()
        with amp.autocast(enabled=True) if is_training else torch.no_grad():
            vlm_input['image_embeds'] = bow_image_conv_encoder(vlm_media)
            result = vlm(**vlm_input, return_dict=True)
            batch_loss = result['loss']

        if losses is not None:
            losses.extend(batch_loss.detach().clone().tolist())
        else:
            losses = batch_loss.detach().clone().tolist()

        if is_training:
            # update the model(s) after processing a batch of episodes
            loss = batch_loss.mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(vlm.parameters(), max_grad_norm)
            torch.nn.utils.clip_grad_norm_(bow_image_conv_encoder.parameters(), max_grad_norm)
            optimizer.step()

        processed_demos += num_demos

        # get statistics for each log interval during training
        if is_training and (processed_demos%log_interval==0 or processed_demos==num_all_demos):
            log_losses_stat(training_status_path, losses, t0, epoch_id, is_training, rd=6)

    # logging the testing loss
    if not is_training:
        log_losses_stat(training_status_path, losses, t0, epoch_id, is_training, rd=6)
    
    gc.collect()
    torch.cuda.empty_cache()

    loss_statistics = get_stat(losses, rd=6, return_type='np')
    return loss_statistics

# Utility functions for checking experiment results
def load_losses(models_dir, exp_name):

    train_loss_fn = "train_loss.npy"
    test_loss_fn = "test_loss.npy"
    model_dir=models_dir + "/" + exp_name

    train_loss = np.load(os.path.join(model_dir, train_loss_fn))
    test_loss = np.load(os.path.join(model_dir, test_loss_fn))

    return train_loss, test_loss

def plot_losses(title, train_loss, test_loss, num_epochs, fig_fp=None):

    epochs = range(1, num_epochs+1)

    plt.plot(epochs,train_loss, label="train loss")
    plt.plot(epochs,test_loss, label="test loss")

    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    if fig_fp:
        plt.savefig(fig_fp)

def test_subgoal_generation(demo, vlm, img_encoder, device, abstract_history, lowlevel_instr_set, tokenizer):

    vlm.to(device)
    img_encoder.to(device)
    vlm.eval()
    img_encoder.eval()
    seed = demo[-1]
    losses = []

    msg = "Proposed Subgoal VS Target Subgoal:\n"

    obss, csg_texts, csg_time_steps, pre_csg_time_steps = prepare_input_seq(
        demo, abstract_history, lowlevel_instr_set)

    csg_texts_tokens = tokenizer(csg_texts, padding=True, return_tensors="pt")
    for key in csg_texts_tokens:
        csg_texts_tokens[key] = csg_texts_tokens[key].to(device)
    csg_tokens_len = csg_texts_tokens['attention_mask'].sum(dim=-1)

    with torch.no_grad():
        img_embeds = img_encoder(obss)

    num_csgs = len(csg_time_steps)

    mission = demo[0][0]['mission']
    vlm_input = tokenizer([f"Goal: {mission}"], max_length=1024, padding="max_length", return_tensors='pt')
    for key in vlm_input:
        vlm_input[key] = vlm_input[key].to(device)
    pre_input_text_len = vlm_input['attention_mask'].sum(dim=-1)[0] # batch size is 1 here

    vlm_input['media_locations'] = torch.zeros(vlm_input['input_ids'].shape, dtype=torch.bool, device=device)

    pre_pre_csg_time_step = -1
    # Calculate the loss of each predicted subgoal by accumulatly preparing the sample input and label
    #   Sample Input:
    #       text: '|image...image|" tokens that correspond to previous subgoal or the intial observation
    #       media: image embedding of the corresponding visual observations
    #   Sample label: tokens of target/completed subgoal
    for idx, pre_csg_time_step in zip(range(num_csgs), pre_csg_time_steps):
        # set up labels for the current subgoal
        num_attended_vis_obss = pre_csg_time_step - pre_pre_csg_time_step
        if abstract_history:
            num_attended_vis_obss = 1
        
        # Setup input ids and attention mask for the sample case of the new subgoal
        #input_text_len = pre_input_text_len + csg_tokens_len[idx]
        media_tokens_len = 2 + num_attended_vis_obss
        pre_subgoal_status_token_len = 0
        pre_subgoal_status_token_len = 3 # subgoal status "[Success]" => "[", "Success", "]"
        '''
        if pre_csg_time_step != 0:
            pre_subgoal_status_token_len = 3 # subgoal status "[Success]" => "[", "Success", "]"
        '''
        added_input_tokens_len = media_tokens_len + pre_subgoal_status_token_len
        input_text_len = pre_input_text_len + added_input_tokens_len
        vlm_input['input_ids'][0, pre_input_text_len:input_text_len] = csg_texts_tokens['input_ids'][idx, :added_input_tokens_len]
        vlm_input['attention_mask'][0, pre_input_text_len:input_text_len] = 1

        # set up media and its locations (that correspond to previous subgoal) that are used to predit the current subgoal
        current_media_locations = [pre_input_text_len]
        if abstract_history:
            vlm_input['image_embeds'] = img_embeds[:, :(idx+1), :, :]
        else:
            # Instead of the 1st 'image', '|' in '|image...image|' corresonds to the 1st media location
            current_media_locations.extend([pre_input_text_len+i for i in range(2, num_attended_vis_obss+1)])
            vlm_input['image_embeds'] = img_embeds[:, :(pre_csg_time_step+1), :, :]
        vlm_input['media_locations'][0, current_media_locations] = True

        # Calculate the loss and update the model
        max_len_generated_sent = 20
        sentence_ending_token = 0 # token 0 is for comma, '!'
        with torch.no_grad():
            generated_tokens = vlm.generate_sentences(
                max_len_generated_sent,
                vlm_input['input_ids'],
                vlm_input['attention_mask'],
                image_embeds = vlm_input['image_embeds'],
                media_locations = vlm_input['media_locations'],
                sentence_ending_token = sentence_ending_token,
                pad_token_id = tokenizer.pad_token_id
                )
        generated_sentence = tokenizer.decode(generated_tokens[0], skip_special_tokens=True)
        context_sentence = tokenizer.decode(vlm_input['input_ids'][0], skip_special_tokens=True)
        completed_subgoal_text = tokenizer.decode(csg_texts_tokens['input_ids'][idx, added_input_tokens_len:csg_tokens_len[idx]], skip_special_tokens=True)
        msg += f"Timestep {pre_csg_time_step}:\n"
        msg += f"\tPrediction Context: {context_sentence}\n"
        msg += f"\tProposed   Subgoal: {generated_sentence}\n"
        msg += f"\tTarget     Subgoal: {completed_subgoal_text}\n"
        
        # Append the target subgoal tokens to input text tokens for assisting the generation of the next subgoal
        input_text_len = pre_input_text_len + csg_tokens_len[idx]
        vlm_input['input_ids'][0, pre_input_text_len:input_text_len] = csg_texts_tokens['input_ids'][idx, :csg_tokens_len[idx]]
        vlm_input['attention_mask'][0, pre_input_text_len:input_text_len] = 1

        # Update some auxilary variables
        pre_input_text_len = input_text_len
        pre_pre_csg_time_step = pre_csg_time_step

    return msg

def test_with_one_demo(demo, model, device, abstract_history, lowlevel_instr_set, tokenizer, skip_label):

    vlm = model['vlm']
    img_encoder = model['img_encoder']
    vlm.to(device)
    img_encoder.to(device)
    vlm.eval()
    img_encoder.eval()

    msg = f"Demo Info:\n"
    msg += f"\tGoal: {demo[0][0]['mission']}\n"
    msg += f"\tCompleted Subgoals In Demo:\n"
    obss, csg_texts, csg_time_steps, pre_csg_time_steps = prepare_input_seq(
        demo, abstract_history, lowlevel_instr_set)
    for count, csg_text in enumerate(csg_texts):
        msg += f"\t[{count+1}] {csg_text}\n"
    msg += "\n"

    demos = [demo]
    dataset = prepare_dataset(device, demos, abstract_history, lowlevel_instr_set, tokenizer, skip_label)
    vlm_input, vlm_media, seed, pre_csg_time_steps = dataset[0]
    with torch.no_grad():
        vlm_input['image_embeds'] = img_encoder(vlm_media)
        result = vlm(**vlm_input, return_dict=True)
        msg += f"Loss: {result['loss'].item()}\n"
    msg += "\n"

    msg += test_subgoal_generation(demo, vlm, img_encoder, device, abstract_history, lowlevel_instr_set, tokenizer)
    return msg

def check_losses(history_type, dataset_size, batch_size, num_epochs, lr, exp_name, models_dir, file):

    train_loss, test_loss = load_losses(models_dir, exp_name)

    title = f"{history_type} History: {dataset_size} demos, batch_size {batch_size}, lr{lr}"
    fig_fn = f"Losses-{history_type}History-{dataset_size}Demos-Batch{batch_size}-{num_epochs}Epochs-LR{lr}"
    fig_fp = models_dir + "/" + exp_name + "/" + fig_fn + ".jpg"
    plot_losses(title, train_loss[:, 0], test_loss[:, 0], num_epochs, fig_fp)

    msg = "[Training and Testing Losses]\n"
    msg += f"\tTrain Loss: {train_loss[:, 0]}\n"
    msg += f"\tTest  Loss: {test_loss[:, 0]}\n"
    print(msg)
    return msg

def load_models(models_dir, exp_name, model_type="recent"):

    model_dir=os.path.join(models_dir, exp_name)

    # the default value of 'model_type' corresponds to 
    # models trained by the most recent epoch
    if model_type == "recent":
        vlm_fn = "vlm.pt"
        img_encoder_fn = "image_conv.pt"
    elif model_type=="best":
        vlm_fn = "vlm_best.pt"
        img_encoder_fn = "image_conv_best.pt"
    elif model_type=="initial":
        vlm_fn = "vlm_init.pt"
        img_encoder_fn = "image_conv_init.pt"
    else:
        raise f"Unsupported model type: {model_type}"
    
    vlm_fp = os.path.join(model_dir, vlm_fn)
    img_encoder_fp = os.path.join(model_dir, img_encoder_fn)

    if os.path.exists(vlm_fp) and os.path.exists(img_encoder_fp):        
        vlm = torch.load(vlm_fp)
        img_encoder =  torch.load(img_encoder_fp)

        model = {"vlm":vlm, "img_encoder":img_encoder}
    else:
        model = None

    return model

def load_three_types_of_models(models_dir, exp_name):

    models = {}

    model_types = ["recent", "best", "initial"]
    for model_type in model_types:
        model = load_models(models_dir, exp_name, model_type)
        if model is not None:
            models[model_type] = model

    return models

def compare_three_types_of_models(demo, models, device, abstract_history, lowlevel_instr_set, tokenizer, skip_label):

    msg = f"\n[Testing 'best', 'recent' and 'initial' models with the demostration with the seed {demo[-1]}]\n"
    for model_type in ['best', 'recent', 'initial']:
        if model_type in models:
            msg += "\n"
            msg += f"=== Testing {model_type} model [Start] ===\n"
            msg += test_with_one_demo(demo, models[model_type], device, abstract_history, lowlevel_instr_set, tokenizer, skip_label)
            msg += f"=== Testing {model_type} model [Finish] ===\n"
    return msg

def check_one_experiment(
    exp_name, abstract_history, dataset_size, batch_size, num_epochs, lr,
    models_dir, test_demo,
    lowlevel_instr_set, tokenizer, skip_label,
    device, file):
    if abstract_history:
        history_type = "Abstract"
    else:
        history_type = "Full"

    msg = check_losses(history_type, dataset_size, batch_size, num_epochs, lr, exp_name, models_dir, file=file)

    models = load_three_types_of_models(models_dir, exp_name)
    msg += "\n"
    msg += compare_three_types_of_models(test_demo, models, device, abstract_history, lowlevel_instr_set, tokenizer, skip_label)
    print(msg, file=file)

# For Testing
# Total number of available unit tests
num_unit_test_case = 2
# Test Cases
# Test Case 1: test if the two implemnetations of loss calculation over a trajectory are correct
def unit_test_loss_cal_by_subgoal_vs_episode(
    test_demo, device, abstract_history,
    lowlevel_instr_set, tokenizer, vlm, bow_image_conv_encoder,
    skip_label
):
    print("===Test Case 1: test if the two implemnetations of loss calculation over a trajectory are correct===")
    # Calculate loss of each subgoal in an episode
    loss_per_csg_calc = calc_loss_per_subgoal(
        device, test_demo, abstract_history,
        lowlevel_instr_set, tokenizer, vlm, bow_image_conv_encoder,
        debug=True, skip_label=skip_label)
    loss_per_csg_calc = loss_per_csg_calc.item()

    # Calculate loss episode by episode
    vlm.to(device)
    bow_image_conv_encoder.to(device)
    vlm.eval()
    bow_image_conv_encoder.eval()
    cri_obs_emb_indices_to_check = range(4)
    dataset = prepare_dataset(
        device, [test_demo], abstract_history,
        lowlevel_instr_set, tokenizer, skip_label)
    vlm_input, vlm_media, seed, pre_csg_time_steps = dataset[0]
    with torch.no_grad():
        image_embeds = bow_image_conv_encoder([vlm_media])
        vlm_input['image_embeds'] = image_embeds
        result = vlm(**vlm_input, return_dict=True)
        loss = result['loss']
    msg_prefix = "[Unit Test: losses calculated episode by episode]"
    unit_test_log_helper(msg_prefix, vlm_input, [seed], [pre_csg_time_steps], [loss], cri_obs_emb_indices_to_check, abstract_history)
    loss_over_demo = loss.item()

    print("[Unit Test: Summary]")
    print(f"\tloss_per_csg_calc:{loss_per_csg_calc}")
    print(f"\tloss_over_demo   :{loss_over_demo}")
    print(f"\tDifference       :{round(loss_per_csg_calc-loss_over_demo, 6)}\n")


# Test Case 2: test if the losses calculated per episode match that calculated by a batch mode
def unit_test_loss_cal_by_episode_vs_batch(
    test_demos, device, abstract_history,
    lowlevel_instr_set, tokenizer, vlm, bow_image_conv_encoder,
    skip_label
):
    print("=== Test Case 2: : test if the losses calculated per episode match that calculated by a batch mode ===")

    # calculate losses of episodes in a batch
    vlm.to(device)
    bow_image_conv_encoder.to(device)
    vlm.eval()
    bow_image_conv_encoder.eval()
    num_demos = len(test_demos)
    cri_obs_emb_indices_to_check = range(4)

    # Calculate the batch loss
    vlm_input, vlm_media, seeds, all_pre_csg_time_steps = prepare_vlm_input_per_batch(
        device, test_demos, abstract_history,
        lowlevel_instr_set, tokenizer, skip_label)
    with torch.no_grad():
        vlm_input['image_embeds'] = bow_image_conv_encoder(vlm_media)
        result = vlm(**vlm_input, return_dict=True)
        losses_batch = result['loss']
    msg_prefix = "[Unit Test: losses calculated in a batch processing mode]"
    cri_obs_emb_indices_to_check = range(4)
    unit_test_log_helper(msg_prefix, vlm_input, seeds, all_pre_csg_time_steps, losses_batch, cri_obs_emb_indices_to_check, abstract_history)

    # Calculate loss episode by episode
    losses_episodes = torch.zeros(num_demos, device=device)
    for idx, test_demo in enumerate(test_demos):
        dataset = prepare_dataset(
            device, [test_demo], abstract_history,
            lowlevel_instr_set, tokenizer, skip_label)
        vlm_input, vlm_media, seed, pre_csg_time_steps = dataset[0]

        with torch.no_grad():
            image_embeds = bow_image_conv_encoder([vlm_media])
            vlm_input['image_embeds'] = image_embeds
            result = vlm(**vlm_input, return_dict=True)
            loss = result['loss']

        msg_prefix = "[Unit Test: losses calculated episode by episode]"
        unit_test_log_helper(msg_prefix, vlm_input, [seed], [pre_csg_time_steps], [loss], cri_obs_emb_indices_to_check, abstract_history)
        losses_episodes[idx] = loss
    
    print("[Unit Test: Summary]")
    print(f"\tlosses_batch   :{losses_batch}")
    print(f"\tlosses_episodes:{losses_episodes}")
    print(f"\tDifference     :{torch.round(losses_batch-losses_episodes, decimals=6)}\n")

# Helper functions for testing
def unit_test_log_helper(msg_prefix, vlm_input, seeds, all_pre_csg_time_steps, losses, cri_obs_emb_indices_to_check, abstract_history):
    input_ids_lens = vlm_input['attention_mask'].sum(-1)
    num_episodes = len(seeds)
    msg = msg_prefix
    for ep_idx, seed, pre_csg_time_steps, input_ids_len, loss in zip(range(num_episodes), seeds, all_pre_csg_time_steps, input_ids_lens, losses):
        msg += f"\nEpisode {ep_idx+1}: Seed {seed}"
        msg += f"\ntime steps when a subgoal completes (exlcuding the last subgoal):"
        msg += f"\n\t{pre_csg_time_steps}"
        msg += f"\ninput_ids ({input_ids_len}):"
        msg += f"\n\t{vlm_input['input_ids'][ep_idx, :input_ids_len]}"
        msg += f"\nmedia_locations:"
        msg += f"\n\t{vlm_input['media_locations'][ep_idx, :input_ids_len]}"
        msg += f"\nlabels:"
        msg += f"\n\t{vlm_input['labels'][ep_idx, :input_ids_len]}"
        msg += f"\ninstance_weights:"
        msg += f"\n\t{vlm_input['instance_weights'][ep_idx, :input_ids_len]}"
        msg += f"\nimage_embeds:"
        if abstract_history:
            msg += f"\n\t{vlm_input['image_embeds'][ep_idx, range(len(pre_csg_time_steps)), 0, :][:, cri_obs_emb_indices_to_check]}"
        else: 
            msg += f"\n\t{vlm_input['image_embeds'][ep_idx, pre_csg_time_steps, 0, :][:, cri_obs_emb_indices_to_check]}"
        msg += f"\nloss: {loss.item()}\n"
    print(msg)

def calc_loss_per_subgoal(
    device, demo, abstract_history,
    lowlevel_instr_set, tokenizer, vlm, bow_image_conv_encoder,
    debug=False, skip_label=-1):

    vlm.to(device)
    bow_image_conv_encoder.to(device)

    if debug:
        vlm.eval()
        bow_image_conv_encoder.eval()

    seed = demo[-1]
    losses = []

    obss, csg_texts, csg_time_steps, pre_csg_time_steps = prepare_input_seq(demo, abstract_history, lowlevel_instr_set)

    csg_texts_tokens = tokenizer(csg_texts, padding=True, return_tensors="pt")
    for key in csg_texts_tokens:
        csg_texts_tokens[key] = csg_texts_tokens[key].to(device)
    csg_tokens_len = csg_texts_tokens['attention_mask'].sum(dim=-1)
    if debug:
        with torch.no_grad():
            img_embeds = bow_image_conv_encoder([obss])
    else:
        with amp.autocast(enabled=True):
            img_embeds = bow_image_conv_encoder([obss])
    num_csgs = len(csg_time_steps)

    mission = demo[0][0]['mission']
    vlm_input = tokenizer([f"Goal: {mission}"], max_length=512, padding="max_length", return_tensors='pt')
    for key in vlm_input:
        vlm_input[key] = vlm_input[key].to(device)
    pre_input_text_len = vlm_input['attention_mask'].sum(dim=-1)[0] # batch size is 1 here

    vlm_input['media_locations'] = torch.zeros(vlm_input['input_ids'].shape, dtype=torch.bool, device=device)

    if debug:
        label_mask_all_csgs = torch.zeros(vlm_input['input_ids'].shape, dtype=torch.bool, device=device)

    pre_pre_csg_time_step = -1
    num_tokens_subgoal_status = 3 # Each of '[Success]', '[Failure]' and '[None]' maps to three tokens, '[', Success/Failure/None, ']'
    # Calculate the loss of each predicted subgoal by accumulatly preparing the sample input and label
    #   Sample Input:
    #       text: '|image...image|" tokens that correspond to previous subgoal or the intial observation
    #       media: image embedding of the corresponding visual observations
    #   Sample label: tokens of target/completed subgoal
    loss = 0
    for idx, pre_csg_time_step in zip(range(num_csgs), pre_csg_time_steps):
        # Setup input ids and attention mask for the sample case of the new subgoal
        input_text_len = pre_input_text_len + csg_tokens_len[idx]
        vlm_input['input_ids'][0, pre_input_text_len:input_text_len] = csg_texts_tokens['input_ids'][idx, :csg_tokens_len[idx]]
        vlm_input['attention_mask'][0, pre_input_text_len:input_text_len] = 1

        # set up labels for the current subgoal
        num_attended_vis_obss = pre_csg_time_step - pre_pre_csg_time_step
        if abstract_history:
            num_attended_vis_obss = 1

        label_start_idx = pre_input_text_len + (2+num_attended_vis_obss) + num_tokens_subgoal_status
        label_mask = torch.zeros(vlm_input['input_ids'].shape, dtype=torch.bool, device=device)
        label_mask[0, label_start_idx:input_text_len] = True
        vlm_input['labels'] = vlm_input['input_ids'].masked_fill(label_mask==0, skip_label)
        if debug:
            label_mask_all_csgs[0, label_start_idx:input_text_len] = True

        # set up media and its locations (that correspond to previous subgoal) that are used to predit the current subgoal
        current_media_locations = [pre_input_text_len]
        if abstract_history:
            vlm_input['image_embeds'] = img_embeds[:, :(idx+1), :, :]
        else:
            # Instead of the 1st 'image', '|' in '|image...image|' corresonds to the 1st media location
            current_media_locations.extend([pre_input_text_len+i for i in range(2, num_attended_vis_obss+1)])
            vlm_input['image_embeds'] = img_embeds[:, :(pre_csg_time_step+1), :, :]
        vlm_input['media_locations'][0, current_media_locations] = True

        # Calculate the loss
        with torch.no_grad() if debug else amp.autocast(enabled=True):
            result = vlm(**vlm_input, return_dict=True)

        loss_csg = result['loss']
        losses.append(loss_csg.item())
        loss += loss_csg
        
        # Update some auxilary variables
        pre_input_text_len = input_text_len
        pre_pre_csg_time_step = pre_csg_time_step

    # the average loss cross all subgoals.
    # It is a tensor that is used to update the model when in training mode
    loss = loss/num_csgs

    if debug:
        labels_all_csgs = vlm_input['input_ids'].masked_fill(label_mask_all_csgs==0, skip_label)
        msg = "[Unit Test - Loss Compulation Per Subgoal]..."
        msg += f"\nMission (seed {seed}): {mission}"
        msg += f"\ntime steps when a subgoal completes:"
        msg += f"\n\t{csg_time_steps}"
        msg += "\ncsg_texts:"
        for csg_text in csg_texts:
            msg += f"\n\t{csg_text}"
        msg += f"\ninput_ids ({vlm_input['attention_mask'].sum(-1)}):"
        msg += f"\n\t{vlm_input['input_ids'][0, :vlm_input['attention_mask'].sum(-1)]}"
        msg += f"\nmedia_locations:"
        msg += f"\n\t{vlm_input['media_locations'][0, :vlm_input['attention_mask'].sum(-1)]}"
        msg += f"\nall labels:"
        msg += f"\n\t{labels_all_csgs[0, :vlm_input['attention_mask'].sum(-1)]}"
        msg += f"\nimage_embeds:"
        if abstract_history:
            msg += f"\n\t{vlm_input['image_embeds'][0, range(len(pre_csg_time_steps)), 0, 0:4]}"
        else:            
            msg += f"\n\t{vlm_input['image_embeds'][0, pre_csg_time_steps, 0, 0:4]}"
        msg += f"\nloss: {loss.item()}\n"
        print(msg)

    del vlm_input
    del csg_texts_tokens
    del img_embeds
    del result
    del losses
    gc.collect()
    torch.cuda.empty_cache()

    return loss