import time
import datetime
import gc
from einops import rearrange
import torch
import torch.nn as nn
from torch.cuda import amp
import numpy as np

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
    
    def forward(self, obss):
        batch_size = 1
        images = self.image_preproc(obss, device=self.device)
        images = images.unsqueeze(dim=0)
        images = rearrange(images, 'b t h w c -> (b t) c h w')
        image_embeds = self.image_conv(images)
        # convert to the shape : (batch_size, times, height*width, feature_dim)
        image_embeds = rearrange(image_embeds, '(b t) d h w-> b t (h w) d', b=batch_size)
        return image_embeds

def format_time(elapsed):
    elapsed_rounded = int(round((elapsed)))
    return str(datetime.timedelta(seconds=elapsed_rounded))

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

# Output:
#   obss: list of (critical) observations.
#       The last observation of a successful trajectory is not saved.
#       The information of a completed subgoal (at time step t) is saved together with the observation at time step (t-1)
#   csg_time_steps: list of time steps when a subgoal completes.
#   csg_texts: list of 'csg_text'
#       csg_text: "completed_subgoal"<image...image>.
#       The # of 'image' indicates the # of elapsed time steps since the last completed subgoal/mission starts.
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

    for t in range(0, total_time_steps):
        completed_subgoals = demo[t][3]

        if len(completed_subgoals):
            # number of elapsed visual observations for completing the previous subgoal 
            num_attended_vis_obss = pre_csg_time_step - pre_pre_csg_time_step
            if abstract_history:
                obs = demo[pre_csg_time_step][0]
                obss.append(obs)
                num_attended_vis_obss = 1

            csg_text = "<"+"image"*num_attended_vis_obss+">"+lowlevel_instr_set.get_completed_subgoals_msg(completed_subgoals)
            csg_texts.append(csg_text)
            csg_time_steps.append(time_step)
            pre_csg_time_steps.append(pre_csg_time_step)
            pre_pre_csg_time_step = pre_csg_time_step
            pre_csg_time_step = time_step

        time_step += 1

    return  obss, csg_texts, csg_time_steps, pre_csg_time_steps

# Helper function for testing
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
            img_embeds = bow_image_conv_encoder(obss)
    else:
        with amp.autocast(enabled=True):
            img_embeds = bow_image_conv_encoder(obss)
    num_csgs = len(csg_time_steps)

    mission = demo[0][0]['mission']
    vlm_input = tokenizer([mission], max_length=512, padding="max_length", return_tensors='pt')
    for key in vlm_input:
        vlm_input[key] = vlm_input[key].to(device)
    pre_input_text_len = vlm_input['attention_mask'].sum(dim=-1)[0] # batch size is 1 here

    vlm_input['media_locations'] = torch.zeros(vlm_input['input_ids'].shape, dtype=torch.bool, device=device)

    if debug:
        label_mask_all_csgs = torch.zeros(vlm_input['input_ids'].shape, dtype=torch.bool, device=device)

    pre_pre_csg_time_step = -1
    # Calculate the loss of each predicted subgoal by accumulatly preparing the sample input and label
    #   Sample Input:
    #       text: '<image...image>" tokens that correspond to previous subgoal or the intial observation
    #       media: image embedding of the corresponding visual observations
    #   Sample label: tokens of target/completed subgoal
    for idx, pre_csg_time_step in zip(range(num_csgs), pre_csg_time_steps):
        # Setup input ids and attention mask for the sample case of the new subgoal
        input_text_len = pre_input_text_len + csg_tokens_len[idx]
        vlm_input['input_ids'][0, pre_input_text_len:input_text_len] = csg_texts_tokens['input_ids'][idx, :csg_tokens_len[idx]]
        vlm_input['attention_mask'][0, pre_input_text_len:input_text_len] = 1

        # set up labels for the current subgoal
        num_attended_vis_obss = pre_csg_time_step - pre_pre_csg_time_step
        if abstract_history:
            num_attended_vis_obss = 1

        label_mask = torch.zeros(vlm_input['input_ids'].shape, dtype=torch.bool, device=device)
        label_mask[0, (pre_input_text_len+(2+num_attended_vis_obss)):input_text_len] = True
        vlm_input['labels'] = vlm_input['input_ids'].masked_fill(label_mask==0, skip_label)
        if debug:
            label_mask_all_csgs[0, (pre_input_text_len+(2+num_attended_vis_obss)):input_text_len] = True

        # set up media and its locations (that correspond to previous subgoal) that are used to predit the current subgoal
        current_media_locations = [pre_input_text_len]
        if abstract_history:
            vlm_input['image_embeds'] = img_embeds[:, :(idx+1), :, :]
        else:
            # Instead of the 1st 'image', '<' in '<image...image>' corresonds to the 1st media location
            current_media_locations.extend([pre_input_text_len+i for i in range(2, num_attended_vis_obss+1)])
            vlm_input['image_embeds'] = img_embeds[:, :(pre_csg_time_step+1), :, :]
        vlm_input['media_locations'][0, current_media_locations] = True

        # Calculate the loss and update the model
        if debug:
            with torch.no_grad():
                result = vlm(**vlm_input, return_dict=True)
        else:
            with amp.autocast(enabled=True):
                result = vlm(**vlm_input, return_dict=True)
        loss_csg = result['loss']
        losses.append(loss_csg.item())
        
        # Update some auxilary variables
        pre_input_text_len = input_text_len
        pre_pre_csg_time_step = pre_csg_time_step
    
    loss = torch.tensor(losses).mean()

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
        msg += f"\nloss: {loss}\n"
        print(msg)

    del vlm_input
    del csg_texts_tokens
    del img_embeds
    del result
    del losses
    gc.collect()
    torch.cuda.empty_cache()

    return loss

# Prepare the vlm input for one demo (successful trajectory) such that
# * call the VLM on the entire token sequence once
# * use attention_mask to facilitate the retrival of each completed subgoal sample
def prepare_vlm_input_per_demo(device, demo, abstract_history, lowlevel_instr_set, tokenizer, skip_label=-1):
    seed = demo[-1]
    obss = []

    obss, csg_texts, csg_time_steps, pre_csg_time_steps = prepare_input_seq(demo, abstract_history, lowlevel_instr_set)
    mission = demo[0][0]['mission']
    input_text_seq = mission + "".join(csg_texts)

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
    
    # save the labels of the last subgoal
    label_masks[0, (media_seq_end+1):tidx] = True
    instance_weights[0, (media_seq_end+1):tidx] = 1.0/(tidx-media_seq_end-1)

    vlm_input['media_locations'] = media_locations
    vlm_input['instance_weights'] = instance_weights

    vlm_input['labels'] = vlm_input['input_ids'].masked_fill(label_masks==0, skip_label)        

    return vlm_input, obss, seed, pre_csg_time_steps

def prepare_dataset(device, demos, abstract_history, lowlevel_instr_set, tokenizer, skip_label=-1):
    result = []
    for demo in demos:
        vlm_input, vlm_media, seed, pre_csg_time_steps = prepare_vlm_input_per_demo(
            device, demo, abstract_history, lowlevel_instr_set, tokenizer, skip_label)
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
    debug=False,
    test_demo_idx=0):

    vlm.to(device)
    bow_image_conv_encoder.to(device)

    msg = ""
    t0 = time.time()
    losses=[]

    demo_ids = np.arange(0, len(demos))
    if debug:
        vlm.eval()
        bow_image_conv_encoder.eval()
        test_demo = [demos[test_demo_idx]]
        dataset = prepare_dataset(
            device, test_demo, abstract_history,
            lowlevel_instr_set, tokenizer, skip_label)
        vlm_input, vlm_media, seed, pre_csg_time_steps = dataset[0]
        vlm_input['image_embeds'] = bow_image_conv_encoder(vlm_media)
        result = vlm(**vlm_input, return_dict=True)
        loss = result['loss']
        msg = "[Unit Test - Loss Compulation Over Trajectory]..."
        msg += f"\nseed: {seed}"
        msg += f"\ntime steps when a subgoal completes:"
        msg += f"\n\t{pre_csg_time_steps}"
        msg += f"\ninput_ids ({vlm_input['attention_mask'].sum(-1)}):"
        msg += f"\n\t{vlm_input['input_ids'][0, :vlm_input['attention_mask'].sum(-1)]}"
        msg += f"\nmedia_locations:"
        msg += f"\n\t{vlm_input['media_locations'][0, :vlm_input['attention_mask'].sum(-1)]}"
        msg += f"\nlabels:"
        msg += f"\n\t{vlm_input['labels'][0, 1:vlm_input['attention_mask'].sum(-1)]}"
        msg += f"\ninstance_weights:"
        msg += f"\n\t{vlm_input['instance_weights'][0, 1:vlm_input['attention_mask'].sum(-1)]}"
        msg += f"\nimage_embeds:"
        if abstract_history:
            msg += f"\n\t{vlm_input['image_embeds'][0, range(len(pre_csg_time_steps)), 0, 0:4]}"
        else:            
            msg += f"\n\t{vlm_input['image_embeds'][0, pre_csg_time_steps, 0, 0:4]}"
        msg += f"\nloss: {loss.item()}\n"
        print(msg)

        # manage memory
        del vlm_media
        del vlm_input
        del result
        gc.collect()
        torch.cuda.empty_cache()
        return loss.item() # only test the first demostration of the input, 'demos'
    else:
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
    if len(demos)%log_interval != 0:
        num_log_intervals += 1

    log_msg(training_status_path, msg)

    for log_interval_idx in range(num_log_intervals):
        start_idx = log_interval_idx*log_interval
        end_idx = (1+log_interval_idx)*log_interval
        if log_interval_idx+1 != num_log_intervals:
            demos_interval = [demos[idx] for idx in demo_ids[start_idx:end_idx]]
        else:
            demos_interval = [demos[idx] for idx in demo_ids[start_idx:]]
        dataset = prepare_dataset(
            device, demos_interval, abstract_history,
            lowlevel_instr_set, tokenizer, skip_label)

        num_demos = len(dataset)
        for demo_idx in range(num_demos):
            vlm_input, vlm_media, seed, pre_csg_time_steps = dataset[demo_idx]
            if is_training:
                # Calculate the training loss
                with amp.autocast(enabled=True):
                    vlm_input['image_embeds'] = bow_image_conv_encoder(vlm_media)
                    result = vlm(**vlm_input, return_dict=True)
                    loss = result['loss']
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
                    loss = result['loss']
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