import argparse
import datetime
import os
import sys

import lightning as L
import pandas as pd
import torch
import yaml
from datasets import load_from_disk
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from knnlm import KNNWrapper

# Define and parse command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str,
    default="configs/main.yaml")
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
def main():
    fabric = L.Fabric(
        accelerator="auto",
        devices="auto",
        # strategy="ddp",
        precision='bf16-mixed')
    fabric.launch()
    fabric.seed_everything(42 + fabric.global_rank)
    
    tokenizer = AutoTokenizer.from_pretrained(
        config['model']['name'])
    
    with fabric.device:
        if 'checkpoint' in config['model']:
            checkpoint_path = config['model']['checkpoint']
            print(f"Loading checkpoint from {checkpoint_path}")
            model = AutoModelForSeq2SeqLM.from_pretrained(
                checkpoint_path)
        else:
            model = AutoModelForSeq2SeqLM.from_pretrained(
                config['model']['name'])
        
        if config['model']['enable_knn']:
            # Injecting KNN
            dimension = model.config.hidden_size
            model = KNNWrapper(
                model=model, 
                dstore_dir = config['knn']['store_dir'], 
                dimension = dimension, 
                no_load_keys = True, 
                move_dstore_to_mem = True, 
                knn_gpu = config['knn']['use_gpu'],
                k = config['knn']['k'], 
                lmbda = config['knn']['lambda'], 
                knn_temp = config['knn']['temperature'], 
                probe = 32)
    
    val_dataset = load_from_disk(config['data']['val_path'])
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset, 
        batch_size = config['data']['eval_batch_size'], 
        shuffle = True)
    
    model = fabric.setup(model)
    val_dataloader = fabric.setup_dataloaders(val_dataloader)
    
    fabric.print("Validating ...")
    date_and_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    save_path = os.path.join(
        "eval", config['model']['name'], 
        date_and_time)
    
    # Save config to save path
    os.makedirs(save_path, exist_ok=True)
    with open(os.path.join(save_path, "config.yaml"), "w") as f:
        yaml.dump(config, f)
    
    model.eval()
    output = []
    part_size = 1024
    part_idx = 0
    bar = tqdm(val_dataloader, desc="Validating")
    for batch in bar:
        batch = preprocess_batch(batch, fabric)
        
        outputs = model.generate(
            input_ids = batch['input_ids'],
            attention_mask = batch['attention_mask'],
            max_new_tokens = 64, 
            num_beams = 5,
            eos_token_id = tokenizer.eos_token_id,
            pad_token_id = tokenizer.eos_token_id)
        
        inputs = tokenizer.batch_decode(batch['input_ids'].detach().cpu().numpy(), skip_special_tokens=True)
        inputs = [inp.split(":")[1].strip() for inp in inputs]
        fabric.print("Model Input:", inputs)
        
        preds = tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True)
        fabric.print("Model Output: ", preds)
        
        batch['labels'][batch['labels'] == -100] = tokenizer.pad_token_id
        labels = tokenizer.batch_decode(batch['labels'].detach().cpu().numpy(), skip_special_tokens=True)
        fabric.print("Target: ", labels)
        
        save_entries = [{
            'input': inp,
            'labels': label,
            'preds': pred}
            for inp, label, pred in zip(inputs, labels, preds)]
        output.extend(save_entries)
        
        if len(output) >= part_size:
            data = pd.DataFrame(output)
            
            fabric.print(f"Saving part {part_idx} for process {fabric.global_rank}...")
            save_file_path = os.path.join(save_path, f"part-{part_idx}-{fabric.global_rank}.jsonl")
            data.to_json(save_file_path, orient='records', lines=True)
            
            output = []
            part_idx += 1
    
    # Save remaining entries
    if len(output) > 0:
        data = pd.DataFrame(output)
        
        fabric.print(f"Saving part {part_idx} for process {fabric.global_rank}...")
        save_file_path = os.path.join(save_path, f"part-{part_idx}-{fabric.global_rank}.jsonl")
        data.to_json(save_file_path, orient='records', lines=True)
    
    fabric.barrier()
    if fabric.global_rank == 0:
        fabric.print("Joining all the parts ...")
        jsonl_files = [f for f in os.listdir(save_path) if f.endswith('.jsonl')]
        jsonl_files.sort()
        
        data = pd.concat([
            pd.read_json(os.path.join(save_path, f), orient='records', lines=True)
            for f in jsonl_files])
        
        save_file_path = os.path.join(save_path, f"eval.jsonl")
        fabric.print("Saving the final output to:", save_file_path)
        data.to_json(save_file_path, orient='records', lines=True)

if __name__ == "__main__":
    torch.set_float32_matmul_precision('medium')
    main()
