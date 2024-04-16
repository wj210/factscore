from factscore.lm import LM
import openai
import sys
import time
import os
import numpy as np
import logging
import concurrent.futures
from functools import partial
sys.setrecursionlimit(10000)  # Increase the recursion limit

MAX_NUM_ERROR = 5

def async_process(fn,inps,workers=10):
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        out = list(executor.map(fn,inps))
    return out

class OpenAIModel(LM):

    def __init__(self, model_name, cache_file=None, key_path="api.key"):
        self.model_name = model_name
        self.key_path = key_path
        self.temp = 0.7
        self.save_interval = 100
        super().__init__(cache_file)

    def load_model(self):
        # load api key
        key_path = self.key_path
        assert os.path.exists(key_path), f"Please place your OpenAI APT Key in {key_path}."
        with open(key_path, 'r') as f:
            api_key = f.readline()
        self.client = openai.OpenAI(api_key = api_key.strip())
        self.model = self.model_name

    def _generate(self, prompt, max_sequence_length=2048, max_output_length=128,few_shot=False):
        if self.add_n % self.save_interval == 0:
            self.save_cache()
        # return a tuple of string (generated text) and metadata (any format)
        # This should be about generating a response from the prompt, no matter what the application is
        if self.model_name == "ChatGPT":
            call_fn = partial(self.call_ChatGPT,max_new_tokens = max_output_length)
        else:
            call_fn = partial(self.call_GPT3,max_tokens = max_output_length)
        if isinstance(prompt,list):
            is_list = True
            message = [[{"role": "user", "content": p}] for p in prompt]
        else:
            is_list = False
            if few_shot and self.model_name == 'ChatGPT':
                message = []
                split_shots = prompt.split('\n\n')
                for i,shot in enumerate(split_shots):
                    split_s = shot.split('\n')
                    message.append({'role':'user','content':split_s[0].strip()})
                    if (i+1) < len(split_shots):
                        message.append({'role':'assistant','content':'\n'.join(split_s[1:]).strip()})
            else:
                message = [{"role": "user", "content": prompt}]
        if is_list:
            out = async_process(call_fn,message,workers = len(message))
        else:
            out = call_fn(message)
        return out 

    def call_ChatGPT(self,message, max_len=1024,max_new_tokens=512):
        # call GPT-3 API until result is provided and then return it
        model_name = "gpt-3.5-turbo-0125"
        response = None
        received = False
        num_rate_errors = 0
        while not received:
            try:
                response = self.client.chat.completions.create(model=model_name,
                                                    messages=message,
                                                    max_tokens = max_new_tokens,
                                                    temperature=self.temp)
                received = True
            except:
                num_rate_errors += 1
                error = sys.exc_info()[0]
                if error == openai.BadRequestError:
                    # something is wrong: e.g. prompt too long
                    logging.critical(f"InvalidRequestError\nPrompt passed in:\n\n{message}\n\n")
                    assert False
                logging.error("API error: %s (%d). Waiting %dsec" % (error, num_rate_errors, np.power(2, num_rate_errors)))
                time.sleep(np.power(2, num_rate_errors))
            if num_rate_errors > MAX_NUM_ERROR:
                return None,None
        return response.choices[0].message.content,response


    def call_GPT3(self,message,max_len=512, max_tokens = 512, num_log_probs=0, echo=False, verbose=False):
        # call GPT-3 API until result is provided and then return it
        model_name='gpt-3.5-turbo-instruct'
        response = None
        received = False
        num_rate_errors = 0
        prompt = message[-1]['content']
        while not received:
            try:
                response = self.client.completions.create(model=model_name,
                                                    prompt=prompt,
                                                    max_tokens = max_tokens,
                                                    temperature=self.temp,
                                                    logprobs=num_log_probs,
                                                    echo=echo)
                received = True
            except:
                error = sys.exc_info()[0]
                num_rate_errors += 1
                if error == openai.BadRequestError:
                    # something is wrong: e.g. prompt too long
                    logging.critical(f"InvalidRequestError\nPrompt passed in:\n\n{prompt}\n\n")
                    assert False
                logging.error("API error: %s (%d)" % (error, num_rate_errors))
                time.sleep(np.power(2, num_rate_errors))
            if num_rate_errors > MAX_NUM_ERROR:
                return None,None
        return response.choices[0].text,response
