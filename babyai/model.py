from typing import List
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.distributions.categorical import Categorical
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import babyai.rl
from babyai.rl.utils.supervised_losses import required_heads
from babyai.utils.format import RawImagePreprocessor

from transformers import top_k_top_p_filtering
from einops import rearrange

# From https://github.com/ikostrikov/pytorch-a2c-ppo-acktr/blob/master/model.py
def initialize_parameters(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.normal_(0, 1)
        m.weight.data *= 1 / torch.sqrt(m.weight.data.pow(2).sum(1, keepdim=True))
        if m.bias is not None:
            m.bias.data.fill_(0)


# Inspired by FiLMedBlock from https://arxiv.org/abs/1709.07871
class FiLM(nn.Module):
    def __init__(self, in_features, out_features, in_channels, imm_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels=in_channels, out_channels=imm_channels,
            kernel_size=(3, 3), padding=1)
        self.bn1 = nn.BatchNorm2d(imm_channels)
        self.conv2 = nn.Conv2d(
            in_channels=imm_channels, out_channels=out_features,
            kernel_size=(3, 3), padding=1)
        self.bn2 = nn.BatchNorm2d(out_features)

        self.weight = nn.Linear(in_features, out_features)
        self.bias = nn.Linear(in_features, out_features)

        self.apply(initialize_parameters)

    def forward(self, x, y):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.conv2(x)
        weight = self.weight(y).unsqueeze(2).unsqueeze(3)
        bias = self.bias(y).unsqueeze(2).unsqueeze(3)
        out = x * weight + bias
        return F.relu(self.bn2(out))


class ImageBOWEmbedding(nn.Module):
   def __init__(self, max_value, embedding_dim):
       super().__init__()
       self.max_value = max_value
       self.embedding_dim = embedding_dim
       self.embedding = nn.Embedding(3 * max_value, embedding_dim)
       self.apply(initialize_parameters)

   def forward(self, inputs):
       offsets = torch.Tensor([0, self.max_value, 2 * self.max_value]).to(inputs.device)
       inputs = (inputs + offsets[None, :, None, None]).long()
       return self.embedding(inputs).sum(1).permute(0, 3, 1, 2)


class ACModel(nn.Module, babyai.rl.RecurrentACModel):
    def __init__(
        self, obs_space, num_of_actions,
        image_dim=128, memory_dim=128, instr_dim=128,
        use_instr=False, lang_model="gru", use_memory=False,
        arch="bow_endpool_res", aux_info=None,
        use_vlm=False, vlm=None, tokenizer=None,
        max_desc_len=0, max_lang_model_input_len=0, max_history_window_vlm=0,
        top_k=50, top_p=0.95, sample_next_token=True):

        super().__init__()

        if use_vlm and vlm is None:
            raise ValueError(f"use_vlm is {use_vlm}, but vlm is {vlm}. Expected a vlm passed in")
        self.use_vlm = use_vlm
        if self.use_vlm:
            # Use the Word Emebdding of the GPT2 model
            self.tokenizer=tokenizer
            self.desc_vocabulary_size = vlm.wte.num_embeddings
            self.desc_embedding_dim   = vlm.wte.embedding_dim
            self.vlm_BOW_Embedding = ImageBOWEmbedding(
                obs_space['image'],
                self.desc_embedding_dim)
            
            # each elem in self.history is a tuple, (Ov, Ol)
            # Ov: (batch, 3, 7, 7)
            # Ol: [instruction_length + description_length], sequence of tokens
            self.history = []
            self.max_history_window_vlm = max_history_window_vlm
            self.max_desc_len = max_desc_len # the maximum length of tokens for a generated sentence at one time step
            self.max_lang_model_input_len = max_lang_model_input_len # the maximum length of tokens the language model and tokenizer can handel
            self.top_k = top_k
            self.top_p = top_p
            self.sample_next_token = sample_next_token

        endpool = 'endpool' in arch
        use_bow = 'bow' in arch
        pixel = 'pixel' in arch
        self.res = 'res' in arch

        # Decide which components are enabled
        self.use_instr = use_instr
        self.use_memory = use_memory
        self.arch = arch
        self.lang_model = lang_model
        self.aux_info = aux_info
        if self.res and image_dim != 128:
            raise ValueError(f"image_dim is {image_dim}, expected 128")
        self.image_dim = image_dim
        self.memory_dim = memory_dim
        self.instr_dim = instr_dim

        self.obs_space = obs_space

        for part in self.arch.split('_'):
            if part not in ['original', 'bow', 'pixels', 'endpool', 'res']:
                raise ValueError("Incorrect architecture name: {}".format(self.arch))

        # if not self.use_instr:
        #     raise ValueError("FiLM architecture can be used when instructions are enabled")
        self.image_conv = nn.Sequential(*[
            *([ImageBOWEmbedding(obs_space['image'], 128)] if use_bow else []),
            *([nn.Conv2d(
                in_channels=3, out_channels=128, kernel_size=(8, 8),
                stride=8, padding=0)] if pixel else []),
            nn.Conv2d(
                in_channels=128 if use_bow or pixel else 3, out_channels=128,
                kernel_size=(3, 3) if endpool else (2, 2), stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            *([] if endpool else [nn.MaxPool2d(kernel_size=(2, 2), stride=2)]),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            *([] if endpool else [nn.MaxPool2d(kernel_size=(2, 2), stride=2)])
        ])
        self.film_pool = nn.MaxPool2d(kernel_size=(7, 7) if endpool else (2, 2), stride=2)

        # Define instruction embedding
        if self.use_instr:
            if self.lang_model in ['gru', 'bigru', 'attgru']:
                self.word_embedding = nn.Embedding(obs_space["instr"], self.instr_dim)
                if self.lang_model in ['gru', 'bigru', 'attgru']:
                    gru_dim = self.instr_dim
                    if self.lang_model in ['bigru', 'attgru']:
                        gru_dim //= 2
                    self.instr_rnn = nn.GRU(
                        self.instr_dim, gru_dim, batch_first=True,
                        bidirectional=(self.lang_model in ['bigru', 'attgru']))
                    if self.use_vlm:
                        # only use unidirectional GRU
                        self.desc_rnn = nn.GRU(
                                self.desc_embedding_dim, gru_dim, batch_first=True,
                                bidirectional=False)

                    self.final_instr_dim = self.instr_dim # affect the output dim of the GRU
                else:
                    kernel_dim = 64
                    kernel_sizes = [3, 4]
                    self.instr_convs = nn.ModuleList([
                        nn.Conv2d(1, kernel_dim, (K, self.instr_dim)) for K in kernel_sizes])
                    self.final_instr_dim = kernel_dim * len(kernel_sizes)

            if self.lang_model == 'attgru':
                self.memory2key = nn.Linear(self.memory_size, self.final_instr_dim)

            num_module = 2
            self.controllers = []
            for ni in range(num_module):
                mod = FiLM(
                    in_features=self.final_instr_dim,
                    out_features=128 if ni < num_module-1 else self.image_dim,
                    in_channels=128, imm_channels=128)
                self.controllers.append(mod)
                self.add_module('FiLM_' + str(ni), mod)

        # Define memory and resize image embedding
        self.embedding_size = self.image_dim
        if self.use_memory:
            self.memory_rnn = nn.LSTMCell(self.image_dim, self.memory_dim)
            self.embedding_size = self.semi_memory_size

        # Define actor's model
        self.actor = nn.Sequential(
            nn.Linear(self.embedding_size, 64),
            nn.Tanh(),
            nn.Linear(64, num_of_actions)
        )

        # Define critic's model
        self.critic = nn.Sequential(
            nn.Linear(self.embedding_size, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

        # Initialize parameters correctly
        self.apply(initialize_parameters)

        # Avoid the impact of the above parameter initialization
        if self.use_vlm:
            self.vlm=vlm
            self.tokenizer=tokenizer

        # Define head for extra info
        if self.aux_info:
            self.extra_heads = None
            self.add_heads()

    def add_heads(self):
        '''
        When using auxiliary tasks, the environment yields at each step some binary, continous, or multiclass
        information. The agent needs to predict those information. This function add extra heads to the model
        that output the predictions. There is a head per extra information (the head type depends on the extra
        information type).
        '''
        self.extra_heads = nn.ModuleDict()
        for info in self.aux_info:
            if required_heads[info] == 'binary':
                self.extra_heads[info] = nn.Linear(self.embedding_size, 1)
            elif required_heads[info].startswith('multiclass'):
                n_classes = int(required_heads[info].split('multiclass')[-1])
                self.extra_heads[info] = nn.Linear(self.embedding_size, n_classes)
            elif required_heads[info].startswith('continuous'):
                if required_heads[info].endswith('01'):
                    self.extra_heads[info] = nn.Sequential(nn.Linear(self.embedding_size, 1), nn.Sigmoid())
                else:
                    raise ValueError('Only continous01 is implemented')
            else:
                raise ValueError('Type not supported')
            # initializing these parameters independently is done in order to have consistency of results when using
            # supervised-loss-coef = 0 and when not using any extra binary information
            self.extra_heads[info].apply(initialize_parameters)

    def add_extra_heads_if_necessary(self, aux_info):
        '''
        This function allows using a pre-trained model without aux_info and add aux_info to it and still make
        it possible to finetune.
        '''
        try:
            if not hasattr(self, 'aux_info') or not set(self.aux_info) == set(aux_info):
                self.aux_info = aux_info
                self.add_heads()
        except Exception:
            raise ValueError('Could not add extra heads')

    @property
    def memory_size(self):
        return 2 * self.semi_memory_size

    @property
    def semi_memory_size(self):
        return self.memory_dim

    def forward(self, obs, memory, instr_embedding=None):
        if self.use_instr and instr_embedding is None:
            instr_embedding = self._get_instr_embedding(obs.instr)
            if self.use_vlm:
                goal_and_desc_embedding = self._get_desc_embedding(obs.desc)
                # concatenate the instruction and description
                # desc_embedding: (batch_size, length, desc_embedding_dim)
                instr_embedding = goal_and_desc_embedding
        if self.use_instr and self.lang_model == "attgru":
            # outputs: B x L x D
            # memory: B x M
            mask = (obs.instr != 0).float()
            # The mask tensor has the same length as obs.instr, and
            # thus can be both shorter and longer than instr_embedding.
            # It can be longer if instr_embedding is computed
            # for a subbatch of obs.instr.
            # It can be shorter if obs.instr is a subbatch of
            # the batch that instr_embeddings was computed for.
            # Here, we make sure that mask and instr_embeddings
            # have equal length along dimension 1.
            mask = mask[:, :instr_embedding.shape[1]]
            instr_embedding = instr_embedding[:, :mask.shape[1]]

            keys = self.memory2key(memory)
            pre_softmax = (keys[:, None, :] * instr_embedding).sum(2) + 1000 * mask
            attention = F.softmax(pre_softmax, dim=1)
            instr_embedding = (instr_embedding * attention[:, :, None]).sum(1)

        x = torch.transpose(torch.transpose(obs.image, 1, 3), 2, 3)

        if 'pixel' in self.arch:
            x /= 256.0
        x = self.image_conv(x)
        if self.use_instr:
            for controller in self.controllers:
                out = controller(x, instr_embedding)
                if self.res:
                    out = out + x
                x = out
        x = F.relu(self.film_pool(x))
        x = x.reshape(x.shape[0], -1)

        if self.use_memory:
            hidden = (memory[:, :self.semi_memory_size], memory[:, self.semi_memory_size:])
            hidden = self.memory_rnn(x, hidden)
            embedding = hidden[0]
            memory = torch.cat(hidden, dim=1)
        else:
            embedding = x

        if hasattr(self, 'aux_info') and self.aux_info:
            extra_predictions = {info: self.extra_heads[info](embedding) for info in self.extra_heads}
        else:
            extra_predictions = dict()

        x = self.actor(embedding)
        dist = Categorical(logits=F.log_softmax(x, dim=1))

        x = self.critic(embedding)
        value = x.squeeze(1)

        return {'dist': dist, 'value': value, 'memory': memory, 'extra_predictions': extra_predictions}

    def _get_instr_embedding(self, instr):
        lengths = (instr != 0).sum(1).long()
        if self.lang_model == 'gru':
            out, _ = self.instr_rnn(self.word_embedding(instr))
            hidden = out[range(len(lengths)), lengths-1, :]
            return hidden

        elif self.lang_model in ['bigru', 'attgru']:
            masks = (instr != 0).float()

            if lengths.shape[0] > 1:
                seq_lengths, perm_idx = lengths.sort(0, descending=True)
                iperm_idx = torch.LongTensor(perm_idx.shape).fill_(0)
                if instr.is_cuda: iperm_idx = iperm_idx.cuda()
                for i, v in enumerate(perm_idx):
                    iperm_idx[v.data] = i

                inputs = self.word_embedding(instr)
                inputs = inputs[perm_idx]

                inputs = pack_padded_sequence(inputs, seq_lengths.data.cpu().numpy(), batch_first=True)

                outputs, final_states = self.instr_rnn(inputs)
            else:
                instr = instr[:, 0:lengths[0]]
                outputs, final_states = self.instr_rnn(self.word_embedding(instr))
                iperm_idx = None
            final_states = final_states.transpose(0, 1).contiguous()
            final_states = final_states.view(final_states.shape[0], -1)
            if iperm_idx is not None:
                outputs, _ = pad_packed_sequence(outputs, batch_first=True)
                outputs = outputs[iperm_idx]
                final_states = final_states[iperm_idx]

            return outputs if self.lang_model == 'attgru' else final_states

        else:
            ValueError("Undefined instruction architecture: {}".format(self.use_instr))
    
    # === functions for generating a text description based on the history of visual and language observations ===
    
    # Input
    #   descs: 2-D tensor of tokens with padding tokens.
    #          Its shape is (batch, self.max_lang_model_input_len)
    # Output
    #   hidden: 2-D tensor of hidden state for next tokens.
    #           Its shape is (batch, the GRU dimention)
    def _get_desc_embedding(self, desc):
        lengths = (desc != 50256).sum(1).long()
        if self.lang_model == 'gru':
            out, _ = self.desc_rnn(self.vlm.wte(desc))
            hidden = out[range(len(lengths)), lengths-1, :]
            return hidden

        else:
            ValueError("Undefined description architecture: {}".format(self.use_instr))

    # Input:
    #   logits: (batch_size, vocabulary_size)
    # Output:
    #   tokens: (batch_size,)
    def logits_to_token(self, logits, top_k=50, top_p=0.95, sample_next_token=True):
        filter = top_k_top_p_filtering(logits, top_k, top_p)
        probabilities = torch.nn.functional.softmax(filter, dim=-1)
        if sample_next_token:
            tokens = torch.multinomial(probabilities, num_samples=1).squeeze(dim=1)
        else:
            tokens = torch.argmax(probabilities, dim=-1)
        return tokens

    # Inputs:
    #   images: (batch, times, channel, height, width)
    #   texts : a list of text, len(texts)==batch_size
    # Output:
    #   encoded_input: a transformers BatchEncoding object (a dict)
    def prepare_vlm_input(self, images, texts):
        device = images.device
        
        encoded_input = self.tokenizer(
            texts,
            return_tensors="pt",
            max_length=self.max_lang_model_input_len, padding="max_length", truncation=True
        )

        batch_size, num_tokens = encoded_input['input_ids'].shape
        media_locations = torch.zeros(encoded_input['input_ids'].shape, dtype=torch.bool)
        for sample_idx in range(batch_size):
            media_start_locs = [ idx for idx in range(num_tokens-2)
                if (encoded_input['input_ids'][sample_idx][idx]==27
                and encoded_input['input_ids'][sample_idx][idx+1]==9060
                and encoded_input['input_ids'][sample_idx][idx+2]==29)
            ]
        for loc in media_start_locs:
            media_locations[sample_idx,loc] = True
        encoded_input['media_locations'] = media_locations
        
        images = rearrange(images, 'b t c h w -> (b t) c h w')

        # images shape: (batch_size*times, channel, height, width)
        # image_embedding shape: (batch_size*times, featture_dim, height, width)
        image_embedding = self.vlm_BOW_Embedding(images)
        # convert to the shape shape: (batch_size, times, height*width, feature_dim)
        image_embedding = rearrange(image_embedding, '(b t) d h w-> b t (h w) d', b=batch_size)
        encoded_input['image_embeds'] = image_embedding

        for key in encoded_input:
            encoded_input[key] = encoded_input[key].to(device)
             
        return encoded_input
    
    # Functionality:
    #   Generate a text description for the current visual observation given the current history
    # Note:
    #   Padding tokens are used in input encoding, so track the last non-masked token
    #   and use its index in the input to track the index of output hidden state of
    #   the next predicted token.
    # Steps:
    #   Call self.vlm to generate the next token
    #   append the new token to the current sequence of tokens
    #   if the new token is 'end of text' or the number of new tokens reaches self.max_desc_len, stop
    #   otherwise, continue to generate a new token
    # Input:
    #   encoded_input: a dict of encoded text with keys
    #       'input_ids', 'attention_mask', 'media_locations', 'image_embeds'
    # Output:
    #   generated_tokens: batch of sequences of tokens where each sequnce represents a text description
    #   generated_sentences: a list of strings where each string is a text description
    def describe_visual_observations(self, encoded_input):
        self.vlm.eval()
        with torch.no_grad():
            input_ids_lens = encoded_input['attention_mask'].sum(dim=1)

            assert not torch.any(input_ids_lens > (self.max_lang_model_input_len - self.max_desc_len))
            
            batch_idx = range(encoded_input['input_ids'].shape[0])
            generated_tokens = None
            for i in range(self.max_desc_len):
                next_token_idx = input_ids_lens + i
                last_nonmasked_token_idx = next_token_idx - 1
                outputs = self.vlm(**encoded_input, return_dict=True)
                # Note: !!!
                # The hidden state representing the next predicted token locates at the index,
                # last_nonmasked_token_idx, in outputs['logits']
                logits = outputs['logits'][batch_idx, last_nonmasked_token_idx, :]
                next_tokens = self.logits_to_token(
                    logits,
                    self.top_k,
                    self.top_p,
                    self.sample_next_token)

                # Append the new token to the sentence and update the attention mask accordingly
                encoded_input['attention_mask'][batch_idx, next_token_idx] = 1
                encoded_input['input_ids'][batch_idx, next_token_idx] = next_tokens
                encoded_input['media_locations'][batch_idx, next_token_idx] = False

                if i == 0:
                    generated_tokens = next_tokens
                else:
                    generated_tokens = torch.cat((generated_tokens, next_tokens), dim=-1)

            generated_sentences = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
            return generated_tokens, generated_sentences

    # Functionality:
    #   prepare input encoding for generate a text description for the current visual observation
    #   call self.describe_visual_observations() to do the text generation
    #   update the self.history by appending the generated text description
    def generate_descs_and_update_histories(self):
        images = None

        texts = copy.deepcopy(self.history[0])
        batch_size = len(texts)
        for i, (Ov, Ol) in enumerate(self.history[1:]): # exclude goals
            ## Ov: (batch_size, times=1, image_embeding_dim=147)
            # Ov: (batch_size, times=1, channel, height, width)
            # Ol: a list of texts, each of which corresponding to one sample in the batch
            if i == 0:
                images = Ov
            else:
                images = torch.cat([images, Ov], dim=1)

            for b_idx in range(batch_size):
                texts[b_idx] = texts[b_idx] + Ol[b_idx]

        # images: (batch_size, times, channel, height, width)
        encoded_input = self.prepare_vlm_input(images, texts)

        # generated_tokens: (batch, self.max_sent_len)
        # generated_sentences:  a llist of strings
        #generated_tokens, generated_sentences = self.describe_visual_observations(encoded_input)
        generated_tokens = self.vlm.generate_sentences(self.max_desc_len,**encoded_input)
        generated_sentences = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)

        # update the history: self.history
        for b_idx in range(batch_size):           
            self.history[-1][1][b_idx] = self.history[-1][1][b_idx] + generated_sentences[b_idx]
        
        return generated_tokens
    
    # Input:
    #   many_obs: a list of raw observations from environments
    def initialize_history_with_goals(self, many_obs):
        self.history.append([])
        for obs in many_obs:
            self.history[0].append("Goal: " + obs['mission'])
        
    # Input:
    #   many_obs: a list of observations
    def update_history(self, preprocessed_obs):
        batch_size = preprocessed_obs.image.shape[0]

        # Remove the oldest one that is at the index 1.
        # The goal is at the index 0, and should not be counted against self.max_history_window_vlm
        if len(self.history) == self.max_history_window_vlm+1:
            self.history.pop(1)
        
        # reshape dimentions of visual observations: (batch, 7, 7, 3) -> (batch, time=1, 147)
        # Ov = rearrange(preprocessed_obs.image, 'b h w d -> b () (h w d)')
        # Ov: (batch_size, times=1, channel, height, width)
        Ov = rearrange(preprocessed_obs.image, 'b h w c -> b () c h w')

        # add the tag '<image>' and the prompt 'Description:' to initialize the text description
        # Node: the tag '<image>' is not passed to the RL agent as the part of the language observation
        Ol = []
        for _ in range(batch_size):
            Ol.append("<image> Description: ")

        self.history.append((Ov, Ol))

    # pass instruction+desc_text as the text information to the RL agent
    # Removing "<image>" before passing the text observation to the RL agent
    def pass_descriptions_to_agent(self):
        encoded_input = self.tokenizer(
                [x+y[7:] for (x, y) in zip(self.history[0], self.history[-1][1])], 
                max_length=self.max_lang_model_input_len, padding="max_length", truncation=True,
                return_tensors='pt')
        return encoded_input['input_ids']


# ACModel Using Flamingo to provide the feature to the policy and value headers
class FlamingoACModel(nn.Module, babyai.rl.ACModel):
    def __init__(
        self, obs_space, num_of_actions,
        arch="bow_endpool_res",
        vlm=None, tokenizer=None,
        max_desc_len=0, max_lang_model_input_len=0, max_history_window_vlm=0,
        top_k=50, top_p=0.95, sample_next_token=True):

        super().__init__()

        self.use_vlm = True

        # Use the Word Emebdding of the GPT2 model
        #self.tokenizer=tokenizer
        self.desc_vocabulary_size = vlm.wte.num_embeddings
        self.desc_embedding_dim   = vlm.wte.embedding_dim
        self.embedding_size = vlm.wte.embedding_dim
            
        # each elem in self.history is a tuple, (Ov, Ol)
        # Ov: (batch, 3, 7, 7)
        # Ol: [instruction_length + description_length], sequence of tokens
        self.history = []
        self.max_history_window_vlm = max_history_window_vlm
        self.max_desc_len = max_desc_len # the maximum length of tokens for a generated sentence at one time step
        self.max_lang_model_input_len = max_lang_model_input_len # the maximum length of tokens the language model and tokenizer can handel
        self.top_k = top_k
        self.top_p = top_p
        self.sample_next_token = sample_next_token

        # Decide which components are enabled
        self.arch = arch
        self.obs_space = obs_space

        for part in self.arch.split('_'):
            if part not in ['original', 'bow', 'pixels', 'endpool', 'res']:
                raise ValueError("Incorrect architecture name: {}".format(self.arch))

        self.image_preproc = RawImagePreprocessor()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.image_conv = nn.Sequential(*[
            ImageBOWEmbedding(obs_space['image'], self.embedding_size),
            nn.Conv2d(
                in_channels=self.embedding_size, out_channels=self.embedding_size,
                kernel_size=(3, 3), stride=1, padding=1),
            nn.BatchNorm2d(self.embedding_size),
            nn.ReLU(),
            nn.Conv2d(in_channels=self.embedding_size, out_channels=self.embedding_size, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(self.embedding_size),
            nn.ReLU(),
        ])

        #self.fc = nn.Linear(self.max_lang_model_input_len*self.embedding_size, self.embedding_size)

        # Define actor's model
        self.actor = nn.Sequential(
            nn.Linear(self.embedding_size, 64),
            nn.Tanh(),
            nn.Linear(64, num_of_actions)
        )

        # Define critic's model
        self.critic = nn.Sequential(
            nn.Linear(self.embedding_size, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

        # Initialize parameters correctly
        self.apply(initialize_parameters)

        # Avoid the impact of the above parameter initialization
        self.vlm=vlm
        self.tokenizer=tokenizer

        self.to(self.device)

    # For each text sequence, added four tokens
    # Prefix: 
    #   <image> : 27(<),  9060(image),    29(>)
    # Postfix:
    #   tokenizer.sep_token
    #
    # Output:
    # encoded_input: a dict that has the attributes,
    #   input_ids (tokens, int) :   (batch_size, max_lang_model_input_len)
    #   attention_mask (0/1)    :   (batch_size, max_lang_model_input_len)
    #   media_locations (bool)  :   (batch_size, max_lang_model_input_len)
    #   input_ids_len (int)     :   (batch_size, 1)
    #   input_ids_len_per_sample_per_seq: List[List[int]]
    def create_text_tokens(self, batch_sentences:List[List[str]], record_subgoal_time_step=False, use_subgoal_desc=False) -> dict:
        if use_subgoal_desc:
            altered_batch_sentences = [
                sample[0]+self.tokenizer.sep_token +
                "".join(["<image>"+sample[i]+self.tokenizer.sep_token for i in range(1, len(sample))]) for sample in batch_sentences
            ]
        else:
            altered_batch_sentences = [
                "".join(["<image>"+sentence+self.tokenizer.sep_token for sentence in sample])
                for sample in batch_sentences
            ]
        
        encoded_input = self.tokenizer(
            altered_batch_sentences,
            max_length=self.max_lang_model_input_len, padding="max_length",
            truncation=True, return_tensors="pt")

        media_locations = torch.zeros(encoded_input['input_ids'].shape, dtype=torch.bool)
        num_samples, num_tokens = encoded_input['input_ids'].shape
        subgoal_indice_per_sample = []
        for i_sample in range(num_samples):
            media_start_locs = [
                idx for idx in range(num_tokens-2) 
                if (encoded_input['input_ids'][i_sample][idx]==27
                    and encoded_input['input_ids'][i_sample][idx+1]==9060
                    and encoded_input['input_ids'][i_sample][idx+2]==29)
            ]
            for loc in media_start_locs:
                media_locations[i_sample,loc] = True

            if record_subgoal_time_step:
                subgoal_indices = []
                if use_subgoal_desc:
                    if len(media_start_locs) > 0:
                        subgoal_indices = [media_start_locs[i]-1 for i in range(1, len(media_start_locs))]
                    last_valid_sep_idx=0
                else:
                    if len(media_start_locs) > 1:
                        subgoal_indices = [media_start_locs[i]-1 for i in range(1, len(media_start_locs))]
                    last_valid_sep_idx=media_start_locs[-1]+3
                # Find the index of the seperator token for the last text sequence in the current sample
                while last_valid_sep_idx < num_tokens:
                    if encoded_input['input_ids'][i_sample][last_valid_sep_idx] == 50256:
                        break
                    else:
                        last_valid_sep_idx += 1   
                subgoal_indices.append(last_valid_sep_idx)
                subgoal_indice_per_sample.append(subgoal_indices)
        
        encoded_input['media_locations'] = media_locations
        input_ids_len = encoded_input['attention_mask'].sum(dim=1)
        for key in encoded_input:
            encoded_input[key] = encoded_input[key].to(self.device)

        # each sample has at least one image    
        # lengths of the original sentences in the batch
        vlm_input = {"encoded_input":encoded_input, "input_ids_len":input_ids_len}
        if record_subgoal_time_step:
            vlm_input['subgoal_indice_per_sample'] = torch.tensor(subgoal_indice_per_sample, device=self.device)

        return vlm_input

    # FixMe: Currently only support only one process/one running envirionment
    # obss: a list of obs object that is a dict of "image" and "mission"
    #       obs['image']    : visual observation from the environment
    #       obs['mission]   : a string
    def forward(self, obss, record_subgoal_time_step=False, use_subgoal_desc=False):
        images = self.image_preproc(obss, device=self.device)
        images = images.unsqueeze(dim=0)
        if use_subgoal_desc:
            # mission description + the following subgoal descripitons
            # the subgoal description at the last time step is empty
            batch_sentences=[[obss[0]['mission']] + [obss[i]['subgoal'] for i in range(1, len(obss)-1)]]
        else:
            batch_sentences = [[obs['mission'] for obs in obss]]
        batch_size = len(batch_sentences) # it is 1 for now.

        # keys: input_ids, attention_mask, media_locations, input_ids_len
        #       *input_ids_len_per_sample_per_seq: List[List[int]]
        #       image_embeds
        vlm_input = self.prepare_vlm_input(images, batch_sentences, record_subgoal_time_step, use_subgoal_desc)

        vlm_output = self.vlm(**vlm_input['encoded_input'], return_dict=True, extract_feature=True)
        # embedding: (b, max_lang_model_input_len, gpt2_embedding_size)
        embedding = vlm_output.last_hidden_state
        #embedding = embedding.masked_fill(encoded_input['attention_mask']==0, 0)

        num_subgoals = [len(subgoal_indices) for subgoal_indices in vlm_input['subgoal_indice_per_sample'] ]


        # Leave the caller of the forward() to decide how to use the returned values
        # Use the values at the index of last unmasked token, or
        # Use the values at the indices of the ending of all subgoals's description
        #
        # dist: (b, max_lang_model_input_len, num_of_actions)
        #   dist.sample(): (b, max_lang_model_input_len)
        # value: (b, max_lang_model_input_len)
        x = self.actor(embedding)
        dist = Categorical(logits=F.log_softmax(x, dim=-1))

        x = self.critic(embedding)
        value = x.squeeze(-1)

        # encoded_input['subgoal_indice_per_sample']: list of lists. The element indicates the time step of each subgoal.
        # batch_size (number of processes) = length of encoded_input['subgoal_indice_per_sample']
        # number of subgoals in each episode i = length of encoded_input['subgoal_indice_per_sample'][0][i]
        result = {'dist': dist, 'value': value}
        if record_subgoal_time_step:
            result['input_ids_len'] = vlm_input['input_ids_len']
            result['subgoal_indice_per_sample'] = vlm_input['subgoal_indice_per_sample']

        return result

    # Inputs:
    #   images: (batch, times, channel, height, width)
    #   texts : a list of text, len(texts)==batch_size
    # Output:
    #   vlm_input: a dictionary of "encoded_input" and "input_ids_len"
    #       encoded_input: a transformers BatchEncoding object (a dict)
    def prepare_vlm_input(self, images, batch_sentences, record_subgoal_time_step=False, use_subgoal_desc=False):
        batch_size = len(batch_sentences)

        vlm_input = self.create_text_tokens(batch_sentences, record_subgoal_time_step, use_subgoal_desc)
        
        images = images.to(self.device)

        images = rearrange(images, 'b t h w c -> (b t) c h w')

        # images shape: (batch_size*times, channel, height, width)
        # image_embedding shape: (batch_size*times, featture_dim, height, width)
        if 'pixel' in self.arch:
            images /= 256.0
        image_embeds = self.image_conv(images)
        # convert to the shape : (batch_size, times, height*width, feature_dim)
        image_embeds = rearrange(image_embeds, '(b t) d h w-> b t (h w) d', b=batch_size)
        vlm_input['encoded_input']['image_embeds'] = image_embeds
             
        return vlm_input
