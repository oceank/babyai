import os
import torch

from .. import utils


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

def load_skill(skill_model_name, budget_steps):
    skill = {}

    skill['model_name'] = skill_model_name
    skill['model'] = load_model(skill['model_name'])

    # load the learned vocab of the skill and use it to tokenize the subgoal
    skill["obss_preprocessor"] = utils.ObssPreprocessor(skill['model_name'])
    skill["budget_steps"] = budget_steps

    return skill
        