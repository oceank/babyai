# CGSUL Branch: <ins>C</ins>onnect <ins>G</ins>oal and <ins>S</ins>tate <ins>U</ins>sing <ins>L</ins>anguage

## Summary
This branch server to train and evaluate a RL agent with a visual-language model (VLM) in the [BabyAI platform](https://github.com/mila-iqia/babyai). Currently, the VLM, FlamingoGPT2, is integrated into the BabyAI platform that is based on the commit, [mila-iqia/babyai@863f352](https://github.com/mila-iqia/babyai/commit/863f3529371ba45ef0148a48b48f5ae6e61e06cc). Changed source files are:
   - babyai/model.py: integrate the VLM into the Actor-Critic model
   - babyai/rl/algo/base.py: handling the observation history for the VLM during training
   - babyai/utils/agent.py: handling the observation history for the VLM during testing
   - babyai/utils/model.py: update the save_model() to avoid saving the history
   - scripts/train_rl.py: update the RL training script to supported the integrated VLM


## Installation

Requirements of Major Packages:
* [Group of packages for BabyAI]
   - Python 3.10+
   - NumPy 1.22.3+
   - PyTorch 1.11.0+
   - OpenAI Gym 0.21.0
   - blosc 1.21.0
* [Group of packages for [FlamingoGPT2](https://github.com/oceank/cgsul)]
   - flamingo-pytorch 0.0.17
   - einops 0.4.1
   - einops-exts 0.0.3
   - vit-pytorch 0.35.2
   - transformers 4.19.4
   - nvidia-apex
   - nvidia-apex-proc
   - matplotb
   - tensorboardX

### Step 1: create a conda environment

Use conda to create an environment (say, cgsul) with all the dependencies by running:

```
git clone -b cgsul git@github.com:oceank/babyai.git
cd babyai
conda env create -f environment.yaml
source activate cgsul
```
### Step 2: install pytorch and nvidia-apex
```
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
conda install nvidia-apex nvidia-apex-proc -c conda-forge
```
### Step 2: install gym-minigrid, babyai and FlamingoGPT2 using in the editable mode

#### gym-minigrid
```
cd ..
git clone -b cgsul git@github.com:oceank/gym-minigrid.git
cd gym-minigrid
pip install --editable .
```

#### babyai
```
cd ../babyai
pip install --editable .
```

#### FlamingoGPT2
```
cd ..
git clone git@github.com:oceank/cgsul.git
cd cgsul
pip install --editable .
```

Finally, [follow these instructions](###babyai-storage-path)


### Step 3: Setup the BabyAI Storage Path

Add this line to `.bashrc` (Linux), or `.bash_profile` (Mac).

```
export BABYAI_STORAGE='/<PATH>/<TO>/<BABYAI>/<REPOSITORY>/<PARENT>'
```

where `/<PATH>/<TO>/<BABYAI>/<REPOSITORY>/<PARENT>` is the folder where you typed `git clone https://github.com/mila-iqia/babyai.git` earlier.

Models and logs will be produced in this directory, in the folders `**models**` and `**logs**` respectively.


## Other useful information are:
- [Codebase Structure](babyai/README.md)
- [Training, Evaluation and Reproducing Baseline Results](scripts/README.md)
- [BabyAI 1.0+ levels](docs/iclr19_levels.md) and [older levels](docs/bonus_levels.md).
