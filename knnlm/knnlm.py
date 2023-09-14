import json
import os
import time
from enum import Enum, auto
from pathlib import Path

import faiss
import faiss.contrib.torch_utils
import numpy as np
import torch
from transformers import PreTrainedModel


class DIST(Enum):
    l2 = auto()
    dot = auto()
    
    @staticmethod
    def from_string(s):
        try:
            return DIST[s.lower()]
        except KeyError:
            raise ValueError()

class KNNWrapper(PreTrainedModel):
    def __init__(self, 
                 model, 
                 dstore_dir, 
                 dimension, 
                 knn_sim_func=None,
                 no_load_keys=False, 
                 move_dstore_to_mem=False, 
                 knn_gpu=True,
                 recompute_dists = False,
                 k=1024, 
                 lmbda=0.25, 
                 knn_temp=1.0, 
                 probe=32):
        self.config_class = model.config_class
        super().__init__(model.config)
        self.model = model
        self.dstore_dir = dstore_dir
        self.dimension = dimension
        self.lmbda = lmbda
        self.k = k
        self.knn_temperature = knn_temp
        self.probe = probe
        self.knn_sim_func = DIST.l2 if knn_sim_func is None else knn_sim_func
        self.no_load_keys = no_load_keys
        self.recompute_dists = recompute_dists
        self.move_dstore_to_mem = move_dstore_to_mem
        self.knn_gpu = knn_gpu and torch.cuda.is_available() and torch.cuda.device_count() > 0
        
        self.keys = None
        self.values = None
        
        dist_type_to_dist_func = {
            DIST.l2: KNNWrapper.l2,
            DIST.dot: KNNWrapper.dotprod,
        }
        self.dist_func = dist_type_to_dist_func[self.knn_sim_func] # l2 or dot product function
        self.reconstruct_index, self.index = self.setup_faiss()
    
    def setup_faiss(self):
        if not self.dstore_dir:
            raise ValueError('Cannot build a datastore without the data.')
        
        start = time.time()
        index_name = os.path.join(self.dstore_dir, 'index.faiss')
        cpu_index = faiss.read_index(index_name, faiss.IO_FLAG_ONDISK_SAME_DIR)
        print(f'Reading datastore took {time.time() - start} s')
        cpu_index.nprobe = self.probe
        
        if self.knn_gpu:
            start = time.time()
            co = faiss.GpuClonerOptions()
            co.useFloat16 = True
            gpu_index = faiss.index_cpu_to_gpu(faiss.StandardGpuResources(), 0, cpu_index, co)
            print(f'Moving index to GPU took {time.time() - start} s')
        else:
            gpu_index = cpu_index
        
        # make_direct_map() allows calling reconstruct(n), 
        # and reconstructing key vectors given their ids
        # currently, this is implemented only for CPU indexes:
        # https://github.com/facebookresearch/faiss/issues/2181
        cpu_index.make_direct_map()
        
        if not self.no_load_keys:
            self.keys = load_memmap_with_metadata(os.path.join(self.dstore_dir, 'keys.npy'))
        self.vals = load_memmap_with_metadata(os.path.join(self.dstore_dir, 'vals.npy'))
        self.vals = torch.from_numpy(self.vals)
        
        # If you wish to load all the keys into memory
        # CAUTION: Only do this if your RAM can handle it!
        if self.move_dstore_to_mem:
            print('Loading to memory...')
            start = time.time()
            
            if not self.no_load_keys:
                del self.keys
                self.keys_from_memmap = load_memmap_with_metadata(os.path.join(self.dstore_dir, 'keys.npy'))
                self.keys = self.keys_from_memmap[:].astype(np.float16)
            
            del self.vals
            vals_from_memmap = load_memmap_with_metadata(os.path.join(self.dstore_dir, 'vals.npy'))
            self.vals = torch.from_numpy(vals_from_memmap[:]).long()
            del vals_from_memmap
            print('Loading to memory took {} s'.format(time.time() - start))
        
        return cpu_index, gpu_index
    
    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        decoder_attention_mask=None,
        cross_attn_head_mask=None,
        use_cache=None,
        encoder_outputs=None,
        **kwargs # needed to bypass checking
    ):
        # cut decoder_input_ids if past is used
        if past_key_values is not None:
            input_ids = input_ids[:, -1:]

        return {
            "decoder_input_ids": input_ids,
            "past_key_values": past_key_values,
            "encoder_outputs": encoder_outputs,
            "attention_mask": attention_mask,
            "head_mask": head_mask,
            "decoder_head_mask": decoder_head_mask,
            "decoder_attention_mask": decoder_attention_mask,
            "cross_attn_head_mask": cross_attn_head_mask,
            "use_cache": use_cache,
        }
    
    def get_encoder(self):
        return self.model.encoder
    
    def _reorder_cache(self, past_key_values, beam_idx):
        return self.model._reorder_cache(past_key_values, beam_idx)
    
    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        if 'output_hidden_states' in kwargs:
            kwargs.pop('output_hidden_states')
        out = self.model(input_ids=input_ids, 
            labels=labels, attention_mask=attention_mask, 
            output_hidden_states=True, **kwargs)
        
        # last layer's hidden states
        if self.model.config.is_encoder_decoder:
            queries = out.decoder_hidden_states[-1]
        else:
            queries = out.hidden_states[-1]

        lm_logits = out.logits
        lm_logits = torch.nn.functional.log_softmax(lm_logits, dim=-1) # (batch, time, vocab)
        batch, time_dim, vocab_size = lm_logits.shape
        
        shift = 0 if self.model.config.is_encoder_decoder else 1
        if labels is None:
            nonpad_mask = torch.cat([
                torch.zeros([batch, time_dim - 1], dtype=torch.bool),
                torch.ones([batch, 1], dtype=torch.bool),
            ], axis=-1)
        else:
            nonpad_mask = torch.cat([
                labels[:, shift:] != -100, 
                torch.zeros([labels.shape[0], shift], dtype=torch.bool).to(labels.device)
            ], axis=-1)
        
        lm_logits = lm_logits[nonpad_mask] # (nonpad, vocab)
        queries = queries[nonpad_mask] # (nonpad, dim)
        
        dists, knns = self.get_knns(queries) # (nonpad batch * time, k)
        if self.recompute_dists:
            knns_vecs = torch.from_numpy(self.keys[knns])
            dists = self.dist_func(queries, knns_vecs) 
        
        neg_dists = -dists
        knn_log_probs, _ = self.knns_to_log_prob(knns, neg_dists, vocab_size)
        
        interpolated_scores = KNNWrapper.interpolate(
            knn_log_probs.to(out.logits.device), lm_logits.to(out.logits.device), self.lmbda) # (nonpad, vocab)
        out.logits[nonpad_mask] = interpolated_scores
        
        return out
    
    def get_knns(self, queries):
        if not self.knn_gpu:
            queries = queries.cpu()
        dists, knns = self.index.search(queries, self.k)
        return dists, knns
    
    def knns_to_log_prob(self, knns, neg_dists, vocab_size):
        probs = torch.nn.functional.softmax(neg_dists / self.knn_temperature, dim=-1)
        vals_at_knns = self.vals[knns].squeeze(-1) # (nonpad batch * time, k)
        knn_log_probs = torch.full(size=(vals_at_knns.shape[:-1] + (vocab_size,)), fill_value=0.0) \
            .scatter_add(dim=-1, index=vals_at_knns, src=probs).log() # (nonpad_batch * time, vocab)
        knn_log_probs = torch.nan_to_num(knn_log_probs, nan=None, neginf=-10000.0)
        return knn_log_probs.to(self.device), vals_at_knns.to(self.device)
    
    @staticmethod
    def l2(query, keys):
        # query: (batch*time, dim)
        # keys:  (batch*time, k, dim)
        # returns: (batch*time, k)
        return torch.sum((query.unsqueeze(-2) - keys)**2, dim=-1)
    
    @staticmethod
    def dotprod(query, keys):
        # query: (batch, beams, dim)
        # keys:  (batch, 1, time, dim)
        # returns: (batch, beams, time)
        return torch.sum((query.unsqueeze(-2) * keys), dim=-1)
    
    @staticmethod
    def interpolate(knn_log_probs, lm_log_probs, lmbda):
        interpolated = torch.logaddexp(
            lm_log_probs + np.log(1 - lmbda), 
            knn_log_probs + np.log(lmbda))
        
        return interpolated

class KNNSaver(PreTrainedModel):
    def __init__(self, model, dstore_size, dstore_dir, dimension):
        self.config_class = model.config_class
        super().__init__(model.config)
        self.model = model
        self.dstore_size = dstore_size
        self.dstore_dir = dstore_dir
        self.dimension = dimension
        self.dstore_idx = 0
        
        print('Saving fp16')
        keys_filename = os.path.join(self.dstore_dir, 'keys.npy')
        vals_filename = os.path.join(self.dstore_dir, 'vals.npy')
        if os.path.exists(keys_filename) and os.path.exists(vals_filename):
            mode = 'r+'
        else:
            mode = 'w+'
            Path(keys_filename).parent.mkdir(parents=True, exist_ok=True)
        
        self.dstore_keys = np.memmap(keys_filename, dtype=np.float16, mode=mode, shape=(self.dstore_size, self.dimension))
        self.dstore_vals = np.memmap(vals_filename, dtype=np.int32, mode=mode, shape=(self.dstore_size, 1))
    
    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        if labels is None:
            raise ValueError('labels must be provided when saving a datastore. Are you using --predict_with_generate by mistake? If so, disable it')
        
        if 'output_hidden_states' in kwargs:
            kwargs.pop('output_hidden_states')
        out = self.model(input_ids=input_ids, 
            labels=labels, attention_mask=attention_mask, 
            output_hidden_states=True, **kwargs)
        
        # Get last layer's hidden states
        if self.model.config.is_encoder_decoder:
            captured_keys = out.decoder_hidden_states[-1]
        else:
            captured_keys = out.hidden_states[-1]
        
        shift = 0 if self.model.config.is_encoder_decoder else 1
        if shift == 1:
            captured_keys = captured_keys[:, :-1]
        captured_keys = captured_keys.flatten(0, 1) # (batch * time, dim)
        captured_values = labels[:, shift:].flatten(0, 1) # (batch * time)
        
        nonpad_mask = (captured_values != -100)
        
        keys = captured_keys[nonpad_mask]
        values = captured_values[nonpad_mask]
        
        return keys, values
    
    def save_batch(self, keys, values):
        batch_time_size = keys.shape[0]
        if self.dstore_idx + batch_time_size > self.dstore_size:
            print(f'Increasing datastore size to {self.dstore_size}')
            self.dstore_size *= 2
            self.dstore_keys = double_memmap_rows(self.dstore_keys)
            self.dstore_vals = double_memmap_rows(self.dstore_vals)
        try:
            self.dstore_keys[self.dstore_idx:(batch_time_size + self.dstore_idx)] = keys.cpu().detach().numpy().astype(np.float16)
            self.dstore_vals[self.dstore_idx:(batch_time_size + self.dstore_idx)] = values.unsqueeze(-1).cpu().detach().numpy().astype(np.int32)
        except ValueError as ex:
            print(f'Error saving datastore with mode {self.dstore_keys.mode}, did you try to save an already existing datastore?')
            print(f'Delete the files {self.dstore_keys.filename} and {self.dstore_vals.filename} and try again')
            raise ex
        self.dstore_idx += batch_time_size
        
    def reduce_dstore_to_size(self):
        self.dstore_keys = set_memmap_rows(self.dstore_keys, self.dstore_idx)
        self.dstore_vals = set_memmap_rows(self.dstore_vals, self.dstore_idx)
        
        save_memmap_metadata(self.dstore_keys)
        save_memmap_metadata(self.dstore_vals)
        
        print('Dataset has:', self.dstore_idx, "keys")

def save_memmap_metadata(memmap_obj, filename=None):
    """Save metadata of a memmap object to a separate file."""
    
    # Get attributes from memmap object
    shape = memmap_obj.shape
    dtype = str(memmap_obj.dtype)
    
    metadata = {
        'shape': shape,
        'dtype': dtype
    }
    
    if filename is None:
        filename = memmap_obj.filename
    
    meta_filename = os.path.splitext(filename)[0] + "_metadata.json"
    
    with open(meta_filename, "w") as meta_file:
        json.dump(metadata, meta_file)

def resize_memmap(mmap_obj, new_shape):
    # Extract the filename and dtype from the memmap object
    filename = mmap_obj.filename
    dtype = mmap_obj.dtype
    
    # Create a temporary memmap with the new shape
    tmp_filename = mmap_obj.filename + '_temp'
    fp_new = np.memmap(tmp_filename, dtype=dtype, mode='w+', shape=new_shape)
    
    # Determine the minimum size between the old and new files for copying
    min_size = min(mmap_obj.size, fp_new.size)
    
    # Copy data from old memmap to new memmap
    fp_new.flat[:min_size] = mmap_obj.flat[:min_size]
    
    # Close the memmap objects to release file handles
    del mmap_obj
    del fp_new
    
    # Rename files
    os.remove(filename)  # Delete the original file
    os.rename(tmp_filename, filename)  # Rename the temp file to the original name
    
    # Remap the variable to the new memmap and return it
    return np.memmap(filename, dtype=dtype, mode='r+', shape=new_shape)

def double_memmap_rows(mmap_obj):
    # Get the current shape of the memmap object
    old_shape = mmap_obj.shape
    
    # Check if the object is at least 2D (has rows to double)
    if len(old_shape) < 2:
        raise ValueError("The provided memmap object must be at least 2D to have rows.")
    
    # Double the number of rows for the new shape
    new_shape = (2 * old_shape[0],) + old_shape[1:]
    
    # Resize the memmap using the previous function
    return resize_memmap(mmap_obj, new_shape)

def set_memmap_rows(mmap_obj, row_length):
    # Get the current shape of the memmap object
    old_shape = mmap_obj.shape
    
    # If the object is 1D, simply change its length
    if len(old_shape) == 1:
        new_shape = (row_length,)
    # If the object is at least 2D, change its row count
    elif len(old_shape) >= 2:
        new_shape = (row_length,) + old_shape[1:]
    else:
        raise ValueError("Unexpected shape for the memmap object.")
    
    # Resize the memmap using the previous function
    return resize_memmap(mmap_obj, new_shape)

def load_memmap_with_metadata(filename):
    """Load memmap using metadata from a separate file."""
    
    # Load metadata
    meta_filename = os.path.splitext(filename)[0] + "_metadata.json"
    with open(meta_filename, "r") as meta_file:
        metadata = json.load(meta_file)

    shape = tuple(metadata['shape'])
    dtype = np.dtype(metadata['dtype'])

    # Load memmap
    fp = np.memmap(filename, dtype=dtype, mode='r', shape=shape)
    
    return fp

def get_directories_in_folder(path='.'):
    return [os.path.join(path, d) for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
