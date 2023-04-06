import os
import datetime
import torch

from .. import utils
from ..levels.verifier import SKILL_DESCRIPTIONS

from babyai.utils.format import RawImagePreprocessor
from babyai.utils.vlm import BowImageConvEncoder
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from FlamingoGPT2.model import FlamingoGPT2
from babyai.model import FACModel

def get_model_dir(model_name):
    return os.path.join(utils.storage_dir(), "models", model_name)


def get_model_path(model_name, model_version="current"):
    return os.path.join(get_model_dir(model_name), f"model_{model_version}.pt")


def load_model(model_name, raise_not_found=True, model_version="current"):
    path = get_model_path(model_name, model_version)
    try:
        if torch.cuda.is_available():
            model = torch.load(path)
        else:
            model = torch.load(path, map_location=torch.device("cpu"))
        model.eval()
        return model
    except FileNotFoundError:
        if raise_not_found:
            raise FileNotFoundError("No model found at {}".format(path))


def save_model(model, model_name, model_version="current"):
    path = get_model_path(model_name, model_version)
    utils.create_folders_if_necessary(path)
    if hasattr(model, 'use_vlm') and hasattr(model, 'history'):
        # The vlm (Flamingo) is used to describe the current state
        # based on the current partial observation and the history
        # using the Flamingo architecture. model.history intends to 
        # store the agent's history during an episode, which is not
        # a part of the model (Flamingo-based Actor-Critic Model).
        # So, do not save model.history to file.
        if model.use_vlm:
            history = model.history
            model.history = []
            torch.save(model, path)
            model.history = history
    else:
        torch.save(model, path)

def retrieve_skill_description(skill_model_name):
    for skill_desc in SKILL_DESCRIPTIONS:
        if skill_desc in skill_model_name:
            return skill_desc
    err_msg = f"The skill {skill_model_name} is not one of the predefined skills, {skill_desc}."
    err_msg += " Please add it to the list of predefined skills."
    raise ValueError(err_msg)
    
    
def load_skill(skill_model_name, budget_steps, model_version):
    skill = {}

    skill['model_name'] = skill_model_name
    skill['model'] = load_model(skill['model_name'], model_version=model_version)

    # load the learned vocab of the skill and use it to tokenize the subgoal
    skill["obss_preprocessor"] = utils.ObssPreprocessor(skill['model_name'], model_version=model_version)
    skill["budget_steps"] = budget_steps
    skill['description'] = retrieve_skill_description(skill_model_name)
    return skill

'''
    Inputs:
        lang_model_name
            gpt2: 12 transformer layers
            distilgpt2: 6 transformer layers
'''
def create_random_hrl_vlm_model(
        env_name, seed, num_high_level_actions,
        skill_arch, skill_instr_arch, max_history_window_vlm, device,
        lang_model_name="distilgpt2", only_attend_immediate_media=True, abstract_history=False,
        max_lang_model_input_len=1024, algo="ppo", args=None):
    # Define model name
    suffix = datetime.datetime.now().strftime("%y-%m-%d-%H-%M-%S")
    algo = algo
    arch = "HRL-Flamingo"
    hist = "full" # full history
    if abstract_history:
        hist = "abs" # abstract
        if args is not None:
            hist += args.abstract_history_type
    attn = "all" # cross attend to all previous medias
    if only_attend_immediate_media:
        attn = "imd" # cross attend to the immediate media
    arch = f"{arch}_{hist}_{attn}"
    mem = "mem"
    skill_arch = f"{skill_arch}_{skill_instr_arch}_{mem}"
    lang_model_train_mode = "FrozenAll"

    model_name_parts = {
        'env': env_name,
        'algo': algo,
        'arch': arch,
        'skill_arch': skill_arch,
        'seed': seed,
        'suffix': suffix}
    model_name = "{env}_{algo}_{arch}_SKILL_{skill_arch}_SEED{seed}_{suffix}".format(**model_name_parts)
    if args is not None:
        model_name_parts['wtype'] = args.episode_weight_type
        model_name_parts['ecoef'] = args.entropy_coef
        model_name_parts['clip']  = args.clip_eps
        model_name_parts['lr']  = args.lr
        model_name_parts['bs']  = args.skill_budget_steps
        model_name_parts['lrst'] = args.lr_scheduling_type
        model_name_parts['lang_model_train_mode'] = args.lang_model_train_mode
        model_name = "{env}_{algo}_{arch}_Lang{lang_model_train_mode}_lr{lr}s{lrst}_wt{wtype}_ec{ecoef}_cl{clip}_SKILL_bs{bs}_{skill_arch}_SEED{seed}_{suffix}".format(**model_name_parts)
        lang_model_train_mode = args.lang_model_train_mode
    print(f"=== Model Name ===")
    print(f"{model_name}")

    # Parameters for VLM
    print(f"=== Initialize a visual-language model, FlamingoGPT2, to assist the agent ===")
    print(f"[Setup] Use VLM to help the anget to explore the grid world")
    print(f"[Setup] Create a tokenizer and {lang_model_name} language model")

    tokenizer = GPT2Tokenizer.from_pretrained(lang_model_name, return_dict=True)
    tokenizer.pad_token = tokenizer.eos_token # pad token
    tokenizer.sep_token = tokenizer.eos_token # sequence separator token

    lang_model = GPT2LMHeadModel.from_pretrained(
        lang_model_name,
        pad_token_id = tokenizer.pad_token_id,
        sep_token_id = tokenizer.sep_token_id)

    lang_model_config = lang_model.config
    dim_lang_embeds = lang_model_config.n_embd
    depth = lang_model_config.n_layer

    print(f"[Setup] Create a visual encoder")
    train_vis_encoder = True
    dim_img_embeds = dim_lang_embeds
    image_preproc = RawImagePreprocessor()
    visual_observation_bow_flat_dim=147
    img_encoder = BowImageConvEncoder(
        visual_observation_bow_flat_dim, dim_img_embeds,
        image_preproc,
        device)

    print(f"[Setup] Create a Flamingo Model")
    vlm = FlamingoGPT2(
        lang_model=lang_model,       # pretrained language model GPT2 with a language header
        dim = dim_lang_embeds,       # dimensions of the embedding
        depth = depth,               # depth of the language model
        # variables below are for Flamingo trainable modules
        heads = 8,                   # attention heads. 8, 4
        ff_mult=4,                   # 4, 2
        dim_head = 64,               # dimension per attention head. 64, 32
        img_encoder = img_encoder,   # plugin your image encoder (this can be optional if you pass in the image embeddings separately, but probably want to train end to end given the perceiver resampler)
        media_token_id = 3,          # the token id representing the [media] or [image]
        cross_attn_every = 3,        # how often to cross attend
        perceiver_num_latents = 64,  # perceiver number of latents. 64, 32
                                    # It should be smaller than the sequence length of the image tokens
        perceiver_depth = 2,         # perceiver resampler depth
        perceiver_num_time_embeds = max_history_window_vlm,#16, 8
        only_attend_immediate_media = only_attend_immediate_media,
        train_vis_encoder = train_vis_encoder,
        lang_model_train_mode = lang_model_train_mode
    )

    print(f"[Setup] Create a Flamingo-based Actor-Critic Model")
    acmodel = FACModel(
        num_of_actions=num_high_level_actions, device=device,
        vlm=vlm, tokenizer=tokenizer, img_encoder=img_encoder,
        max_lang_model_input_len=max_lang_model_input_len,)

    print_details = True
    print_trainable_parameters(acmodel, print_details=print_details)

    return acmodel, model_name

def print_trainable_parameters(model, print_details=False, gpt_lm_head=False):
    """
    Prints the number of trainable parameters in the model.
    gpt_lm_head: assume the imported language model is a GPT2LMHeadModel
        if True, the lm_head layer is used during forward pass. Then, the gradient of loss with respect to the lm_head layer is computed. So, count these parameters.
        if False, the lm_head layer is not used during forward pass. Then, the gradient of loss with respect to the lm_head layer is not computed. So, do not count these parameters.
    """
    trainable_params = 0
    all_param = 0
    if print_details:
        trainable_params_dict = {}
    for name, param in model.named_parameters():
        if not gpt_lm_head and ("lm_head" in name):
            print(f"Skip counting the parameters of {name} since it is not used during the forward pass.")
            continue

        num_params = param.numel()
        # if using DS Zero 3 and the weights are initialized empty
        if num_params == 0 and hasattr(param, "ds_numel"):
            num_params = param.ds_numel

        all_param += num_params
        if param.requires_grad:
            trainable_params += num_params
            if print_details:
                trainable_params_dict[name] = num_params
    msg = f"\ttrainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    print(msg)
    if print_details:
        for name, num_params in trainable_params_dict.items():
            print(f"\t\t{name}: {num_params}")