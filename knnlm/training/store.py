import argparse
import os
import sys
import time

import faiss
import lightning as L
import numpy as np
import torch
import yaml
from datasets import load_from_disk
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from knnlm import (KNNSaver, get_directories_in_folder,
                   load_memmap_with_metadata, save_memmap_metadata)

# Define and parse command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str,
    default="configs/train-debug.yaml")
args = parser.parse_args()

def load_config(yaml_file_path):
    with open(yaml_file_path, 'r') as yaml_file:
        config = yaml.safe_load(yaml_file)
    return config

# Load the configuration from the YAML file
config = load_config(args.config)
print(config)

def preprocess_batch(model_inputs, fabric):
    input_ids = torch.stack(model_inputs['input_ids'], dim=-1)
    
    # Truncate the input and target sequences to the maximum sequence length
    longest_seq_length = (input_ids != 0).sum(dim=1).max().item()
    input_ids = input_ids[:, :longest_seq_length]
    
    attention_mask = torch.stack(model_inputs['attention_mask'], dim=-1)
    attention_mask = attention_mask[:, :longest_seq_length]
    
    labels = torch.stack(model_inputs['labels'], dim=-1)
    longest_seq_length = (labels != -100).sum(dim=1).max().item()
    labels = labels[:, :longest_seq_length].contiguous()
    
    input_ids, attention_mask, labels,  = fabric.to_device((input_ids, attention_mask, labels))
    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': labels
    }

@torch.inference_mode()
def store():
    fabric = L.Fabric(
        accelerator="auto",
        devices="auto",
        strategy="ddp",
        precision='bf16-mixed')
    fabric.launch()
    fabric.seed_everything(42 + fabric.global_rank)
    
    with fabric.device:
        if 'checkpoint' in config['model']:
            checkpoint_path = config['model']['checkpoint']
            print(f"Loading checkpoint from {checkpoint_path}")
            model = AutoModelForSeq2SeqLM.from_pretrained(
                checkpoint_path)
        else:
            model = AutoModelForSeq2SeqLM.from_pretrained(
                config['model']['name'])
        
        # Injecting KNN
        date = time.strftime("%Y-%m-%d_%H-%M-%S")
        base_store_dir = os.path.join("store", f"{config['model']['name']}", date)
        os.makedirs(base_store_dir, exist_ok=True)
        
        # Save config to save path
        os.makedirs(base_store_dir, exist_ok=True)
        with open(os.path.join(base_store_dir, "config.yaml"), "w") as f:
            yaml.dump(config, f)
        
        store_dir = os.path.join(base_store_dir, f"part-{fabric.global_rank}")
        dimension = model.config.hidden_size
        model = KNNSaver(
            model = model, 
            dstore_size = config['knn']['store_size'] // fabric.world_size, 
            dstore_dir = store_dir, 
            dimension = dimension)
    
    fabric.barrier()
    dataset = load_from_disk(config['data']['train_path'])
    dataloader = torch.utils.data.DataLoader(
        dataset, 
        batch_size = config['data']['train_batch_size'], 
        shuffle = True)
    
    model = fabric.setup(model)
    dataloader = fabric.setup_dataloaders(dataloader)
    
    fabric.print("Generating Embeddings")
    model.eval()
    bar = tqdm(dataloader, desc="Saving dstore")
    for batch in bar:
        batch = preprocess_batch(batch, fabric)
        keys, values = model(**batch)
        model.save_batch(keys, values)
        bar.set_postfix({"Num Keys": len(keys), "Num Values": len(values)})    
    model.reduce_dstore_to_size()
    
    fabric.barrier()
    fabric.print("Building index")
    if fabric.global_rank == 0:
        print("Base Store Dir: ", base_store_dir)
        # Combine all the shards
        store_folders = get_directories_in_folder(base_store_dir)
        
        shard_keys_filenames = [os.path.join(folder, "keys.npy") for folder in store_folders]
        shard_vals_filenames = [os.path.join(folder, "vals.npy") for folder in store_folders]
        
        keys_output_filename = os.path.join(base_store_dir, 'keys.npy')
        vals_output_filename = os.path.join(base_store_dir, 'vals.npy')
        
        concatenated_keys = concatenate_memmaps(
            shard_keys_filenames, keys_output_filename)
        
        concatenated_vals = concatenate_memmaps(
            shard_vals_filenames, vals_output_filename)
        
        print('Concatenated keys shape', concatenated_keys.shape)
        print('Concatenated vals shape', concatenated_vals.shape)
        
        index_name = os.path.join(base_store_dir, f'index.faiss')
        build_index(index_name, concatenated_keys, concatenated_vals)

def concatenate_memmaps(filenames, output_filename):
    # Determine the total size of the concatenated memmap
    current_memmap = load_memmap_with_metadata(filenames[0])
    dtype = current_memmap.dtype
    
    total_size = sum([load_memmap_with_metadata(filename).shape[0] for filename in filenames])
    final_shape = list(current_memmap.shape)
    final_shape[0] = total_size
    final_shape = tuple(final_shape)
    
    # Create the output memmap with the determined size
    concatenated_memmap = np.memmap(output_filename, dtype=dtype, mode='w+', shape=final_shape)
    
    # Copy data from the input memmaps to the output memmap
    start_idx = 0
    for filename in filenames:
        current_memmap = load_memmap_with_metadata(filename)
        concatenated_memmap[start_idx:start_idx + current_memmap.shape[0]] = current_memmap
        start_idx += current_memmap.shape[0]
    
    save_memmap_metadata(concatenated_memmap)
    return concatenated_memmap

def build_index(index_name, keys, vals, num_keys_to_add_at_a_time=1000000, 
    ncentroids=4096, seed=1, code_size=64, probe=32):
        print('Building index')
        ncentroids = min(ncentroids, keys.shape[0] // 40)
        
        dstore_size, dimension = keys.shape
        
        # Initialize faiss index
        quantizer = faiss.IndexFlatL2(dimension)
        index = faiss.IndexIVFPQ(quantizer, dimension,
            ncentroids, code_size, 8)
        index.nprobe = probe
        
        print('Training Index')
        np.random.seed(seed)
        random_sample = np.random.choice(np.arange(vals.shape[0]), size=[min(1000000, vals.shape[0])], replace=False)
        start = time.time()
        # Faiss does not handle adding keys in fp16 as of writing this.
        index.train(keys[random_sample].astype(np.float32))
        print(f'Training took {time.time() - start} s')
        
        print('Adding Keys')
        start = 0
        start_time = time.time()
        while start < dstore_size:
            end = min(dstore_size, start + num_keys_to_add_at_a_time)
            to_add = keys[start:end].copy()
            index.add_with_ids(torch.tensor(to_add.astype(np.float32)), torch.arange(start, end))
            start += num_keys_to_add_at_a_time
            
            if (start % 1000000) == 0:
                print(f'Added {start} tokens so far')
                print(f'Writing Index {start}')
                faiss.write_index(index, f'{index_name}')
        
        print(f'Adding total {start} keys')
        print(f'Adding took {time.time() - start_time} s')
        print(f'Writing Index to {index_name}')
        start = time.time()
        faiss.write_index(index, f'{index_name}')
        print(f'Writing index took {time.time() - start} s')

if __name__ == "__main__":
    torch.set_float32_matmul_precision('medium')
    store()
