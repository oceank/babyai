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


def get_model_path(model_name):
    return os.path.join(get_model_dir(model_name), "model.pt")


def load_model(model_name, raise_not_found=True):
    path = get_model_path(model_name)
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


def save_model(model, model_name):
    path = get_model_path(model_name)
    utils.create_folders_if_necessary(path)
    if model.use_vlm: # Do not save the history to the model file
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
    
    
def load_skill(skill_model_name, budget_steps):
    skill = {}

    skill['model_name'] = skill_model_name
    skill['model'] = load_model(skill['model_name'])

    # load the learned vocab of the skill and use it to tokenize the subgoal
    skill["obss_preprocessor"] = utils.ObssPreprocessor(skill['model_name'])
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
        skill_arch, instr_arch, max_history_window_vlm, device,
        lang_model_name="distilgpt2", only_attend_immediate_media=True):
    # Define model name
    suffix = datetime.datetime.now().strftime("%y-%m-%d-%H-%M-%S")
    algo = "ppo"
    arch = "hrl-flamingo"
    instr = instr_arch
    mem = "mem" 

    model_name_parts = {
        'env': env_name,
        'algo': algo,
        'arch': arch,
        'skill_arch': skill_arch,
        'instr': instr,
        'mem': mem,
        'seed': seed,
        'suffix': suffix}
    model_name = "{env}_{algo}_{arch}_SKILL_{skill_arch}_{instr}_{mem}_SEED_{seed}_{suffix}".format(**model_name_parts)
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
        pad_token_id=tokenizer.pad_token_id,
        sep_token_id = tokenizer.sep_token_id)

    lang_model_config = lang_model.config
    dim_lang_embeds = lang_model_config.n_embd
    depth = lang_model_config.n_layer

    # first take your trained image encoder and wrap it in an adapter that returns the image embeddings
    # here we use the ViT from the vit-pytorch library
    print(f"[Setup] Create a visual encoder using ViT")
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
        only_attend_immediate_media = only_attend_immediate_media
    )

    print(f"[Setup] Create a Flamingo-based Actor-Critic Model")
    acmodel = FACModel(num_of_actions=num_high_level_actions, device=device, vlm=vlm, tokenizer=tokenizer, img_encoder=img_encoder)

    return acmodel, model_name