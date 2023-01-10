import os
import time
import datetime
import gc
from einops import rearrange
import torch
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
# Note:
# The format of input sequence to the VLM:
#   Goal: 'mission'<image>Subgoal 1: 'sg1'<image...image>Subgoal 1 Status: 'status'.Subgoal 2: 'sg2'...
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
    input_text_seq = f"Goal: {mission}" + "".join(csg_texts)

    max_token_seq_len = 512
    vlm_input = tokenizer(input_text_seq, max_length=max_token_seq_len, padding="max_length", return_tensors='pt')
    input_token_seq_len = vlm_input['attention_mask'].sum(dim=-1)[0] # batch size is 1 here
    for key in vlm_input:
        vlm_input[key] = vlm_input[key].to(device)

    # Compute media_locations, labels, instance_weights
    media_locations = torch.zeros(vlm_input['input_ids'].shape, dtype=torch.bool, device=device)
    label_masks = torch.zeros(vlm_input['input_ids'].shape, dtype=torch.bool, device=device)
    instance_weights = torch.zeros(vlm_input['input_ids'].shape, dtype=torch.float, device=device)

    media_seq_open = False
    media_seq_end = None
    # |image|: 91,  9060,  91
    # Subgoal Status: '[Success]', '[Failure]'
    num_tokens_sugboal_status = 3
    tidx = 0
    while tidx < input_token_seq_len:
        if vlm_input['input_ids'][0, tidx] == 91: # '|'
            if not media_seq_open:
                # media_seq_end==None corresponds to the indentification of the first image
                # That does not have any subgoal completed before it.
                if media_seq_end is not None:
                    # store the labels of the passed text section
                    label_start_indice = media_seq_end+1+num_tokens_sugboal_status
                    label_masks[0, label_start_indice:tidx] = True
                    instance_weights[0, label_start_indice:tidx] = 1.0/(tidx-label_start_indice)
                media_locations[0, tidx] = True

                # by pass the first 'image' token whose media location is being taken charge by
                # the prefix token, '|'
                tidx += 1
                media_seq_open = True
            else:
                media_seq_end = tidx
                media_seq_open = False
        elif vlm_input['input_ids'][0, tidx] == 9060: # 'image'
            media_locations[0, tidx] = True

        tidx += 1

    # save the labels of the last subgoal
    label_start_indice = media_seq_end+1+num_tokens_sugboal_status
    label_masks[0, label_start_indice:tidx] = True
    instance_weights[0, label_start_indice:tidx] = 1.0/(tidx-label_start_indice)

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
    test_demo_idx=0,
    batch_size=1):

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
        msg += f"\n\t{vlm_input['labels'][0, :vlm_input['attention_mask'].sum(-1)]}"
        msg += f"\ninstance_weights:"
        msg += f"\n\t{vlm_input['instance_weights'][0, :vlm_input['attention_mask'].sum(-1)]}"
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

    num_batches = len(demos)//batch_size
    if len(demos)%batch_size != 0:
        num_batches += 1

    log_msg(training_status_path, msg)

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

        batch_loss = 0
        num_demos = len(dataset)
        if is_training:
            optimizer.zero_grad()
        for demo_idx in range(num_demos):
            vlm_input, vlm_media, seed, pre_csg_time_steps = dataset[demo_idx]
            if is_training:
                # Calculate the training loss
                with amp.autocast(enabled=True):
                    vlm_input['image_embeds'] = bow_image_conv_encoder(vlm_media)
                    result = vlm(**vlm_input, return_dict=True)
                    loss = result['loss']
                    batch_loss += loss
            else:
                # Calculate the testing loss
                with torch.no_grad():
                    vlm_input['image_embeds'] = bow_image_conv_encoder(vlm_media)
                    result = vlm(**vlm_input, return_dict=True)
                    loss = result['loss']

            losses.append(loss.item())

        if is_training:
            # update the model(s) after processing a batch of episodes
            batch_loss /= num_demos
            batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(vlm.parameters(), max_grad_norm)
            torch.nn.utils.clip_grad_norm_(bow_image_conv_encoder.parameters(), max_grad_norm)
            optimizer.step()

        processed_demos += num_demos

        # get statistics for each log interval during training
        if is_training and (processed_demos%log_interval==0 or processed_demos==len(demos)):
            log_losses_stat(training_status_path, losses, t0, epoch_id, is_training)
    
    # logging the testing loss
    if not is_training:
        log_losses_stat(training_status_path, losses, t0, epoch_id, is_training)
    
    gc.collect()
    torch.cuda.empty_cache()

    loss_statistics = get_stat(losses, rd=4, return_type='np')
    return loss_statistics

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
    #       text: '<image...image>" tokens that correspond to previous subgoal or the intial observation
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
            # Instead of the 1st 'image', '<' in '<image...image>' corresonds to the 1st media location
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