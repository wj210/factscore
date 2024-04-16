import pickle
import os
import time

class LM(object):

    def __init__(self, cache_file):
        self.cache_file = cache_file
        self.cache_dict = self.load_cache()
        self.model = None
        self.add_n = 0

    def load_model(self):
        # load the model and put it as self.model
        raise NotImplementedError()

    def generate(self, prompt, sample_idx=0, max_sequence_length=2048, max_output_length=128,few_shot=False):
        if self.model is None:
            self.load_model()
                
        if isinstance(prompt,list):
            prompt = [p.strip() for p in prompt]
        else:
            prompt = prompt.strip() # it's important not to end with a whitespace
        
        if isinstance(prompt,list):
            cache_keys = []
            out = [None for _ in prompt]
            for i,p in enumerate(prompt):
                cache_key = f"{p}_{sample_idx}"
                if cache_key in self.cache_dict:
                    out[i] = self.cache_dict[cache_key]
                else:
                    cache_keys.append(cache_key)
            ## Find those that are not cached, and generate them, record down the positions
            non_cached_pos = [i for i,g in enumerate(out) if g is None]
            if len(non_cached_pos) == 0:
                return out
            prompt = [prompt[j] for j in non_cached_pos]

            if 'True or False?' in prompt[0]:
                generated = self._generate(prompt, max_sequence_length=max_sequence_length, max_output_length=1)
            else:
                generated = self._generate(prompt, max_sequence_length=max_sequence_length, max_output_length=max_output_length)

            for k,g in enumerate(generated):
                if g[0] != None:
                    self.cache_dict[cache_keys[k]] = g
                    self.add_n += 1
            assert len(generated) == len(non_cached_pos), f"Generated: {len(generated)}, Non-cached: {len(non_cached_pos)}"
            for i,pos in enumerate(non_cached_pos):
                out[pos] = generated[i]
            return out
        else:
            cache_key = f"{prompt}_{sample_idx}"
            if cache_key in self.cache_dict:
                return self.cache_dict[cache_key]
            if 'True or False?' in prompt:
                generated = self._generate(prompt, max_sequence_length=max_sequence_length, max_output_length=1)
            else:
                generated = self._generate(prompt, max_sequence_length=max_sequence_length, max_output_length=max_output_length,few_shot=few_shot)# only fact generation need to account for this.
            
            if generated[0] != None:
                self.cache_dict[cache_key] = generated
                self.add_n += 1
            return generated
        
    def save_cache(self):
        if self.add_n == 0:
            return

        # load the latest cache first, since if there were other processes running in parallel, cache might have been updated
        for k, v in self.load_cache().items():
            self.cache_dict[k] = v

        with open(self.cache_file, "wb") as f:
            pickle.dump(self.cache_dict, f)

    def load_cache(self, allow_retry=True):
        if os.path.exists(self.cache_file):
            while True:
                try:
                    with open(self.cache_file, "rb") as f:
                        cache = pickle.load(f)
                    break
                except Exception:
                    if not allow_retry:
                        assert False
                    print ("Pickle Error: Retry in 5sec...")
                    time.sleep(5)        
        else:
            cache = {}
        return cache



