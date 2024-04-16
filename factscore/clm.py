# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import math
import time
import json
import numpy as np
import torch
from tqdm import tqdm
from collections import defaultdict

from transformers import AutoModelForCausalLM,AutoTokenizer

from factscore.utils import convert_model_to_int8_on_gpu
from factscore.lm import LM

class CLM(LM):
    def __init__(self, model_name, model_dir, cache_file=None):
        self.model_name = model_name
        self.model_dir = model_dir
        if cache_file:
            super().__init__(cache_file)

    def load_model(self):
        self.model = AutoModelForCausalLM.from_pretrained(self.model_dir)
        self.model = convert_model_to_int8_on_gpu(self.model, device='cuda').eval()
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir,padding_size = 'left') # padding_size = 'left' is important for generation
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    def _generate(self, prompts, max_sequence_length=2048, max_output_length=128,
                  end_if_newline=False, end_if_second_newline=False):
        is_single = type(prompts)==str
        if is_single:
            prompts = [prompts]
        max_input_len = max_sequence_length - max_output_length
        tokenized_inputs = self.tokenizer(prompts,padding='longest',truncation=True,max_length = max_input_len,return_tensors='pt')['input_ids'].cuda()
        with torch.no_grad():
            gen_outputs = self.model.generate(
                    tokenized_inputs,
                    max_length=tokenized_inputs.shape[1]+max_output_length,
                    return_dict_in_generate=True,
                    output_scores=True
                )
        
        gen_tokens = gen_outputs["sequences"]
        # saving the logits for the very first token
        gen_scores = gen_outputs["scores"][0].detach().cpu().numpy() # take 1st position, return (B,V) tensor
        gen = self.tokenizer.batch_decode(gen_tokens[:, tokenized_inputs.shape[-1]:])
        
        for g in gen:
            if end_if_newline:
                g = g.split("\n")[0].strip()
            elif end_if_second_newline:
                g = "\n".join(g.split("\n")[:2]).strip()
            
            if self.model_name.startswith("llama-sni"):
                g = g.split("</s>")[0]
        
        assert len(gen)==len(prompts)==gen_scores.shape[0]
        
        out = [(g, s) for g,s in zip(gen,gen_scores)]
        return out

        # generations = []
        # scores = []
        # for curr_input_ids in input_ids:
        #     if len(curr_input_ids) > max_sequence_length - max_output_length:
        #         curr_input_ids = curr_input_ids[-(max_sequence_length - max_output_length):]
        #     curr_input_ids = torch.LongTensor([curr_input_ids]).cuda()
        #     gen_outputs = self.model.generate(
        #         curr_input_ids,
        #         max_length=curr_input_ids.shape[1]+max_output_length,
        #         return_dict_in_generate=True,
        #         output_scores=True
        #     )
        #     gen_tokens = gen_outputs["sequences"]
        #     # saving the logits for the very first token
        #     gen_scores = gen_outputs["scores"][0][0].detach().cpu().numpy()
        #     gen = self.tokenizer.decode(gen_tokens[0, curr_input_ids.shape[-1]:])

        #     if end_if_newline:
        #         gen = gen.split("\n")[0].strip()
        #     elif end_if_second_newline:
        #         gen = "\n".join(gen.split("\n")[:2]).strip()

        #     if verbose and len(generations)==0:
        #         print ("Input:", prompts[0])
        #         print ("Prediction:", gen)

        #     if self.model_name.startswith("llama-sni"):
        #         gen = gen.split("</s>")[0]
                
        #     generations.append(gen)
        #     scores.append(gen_scores)

        # assert len(generations)==len(prompts)==len(scores)
        # if is_single:
        #     return generations[0], scores[0]
        
        # return generations, scores

