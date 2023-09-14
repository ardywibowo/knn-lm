import argparse
import os
import shutil
import sys
import time

import lightning as L
import torch
import yaml
from datasets import load_from_disk
from lightning.fabric.strategies import DeepSpeedStrategy
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

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

def main():
    # DeepSpeed
    ds_config = {
        **config['deepspeed'],
        "train_micro_batch_size_per_gpu": config['data']['train_batch_size'],
        "gradient_accumulation_steps": config['training']['gradient_accumulation_steps']}
    
    fabric = L.Fabric(
        accelerator="auto",
        devices="auto",
        strategy=DeepSpeedStrategy(config=ds_config),
        precision='bf16-mixed')    
    fabric.launch()
    fabric.seed_everything(42 + fabric.global_rank)
    
    # Load pretrained model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        config['model']['name'])
    
    with fabric.device:
        if 'checkpoint' in config['model']:
            checkpoint_path = config['model']['checkpoint']
            print(f"Loading checkpoint from {checkpoint_path}")
            model = AutoModelForSeq2SeqLM.from_pretrained(
                checkpoint_path, torch_dtype=torch.bfloat16)
        else:
            model = AutoModelForSeq2SeqLM.from_pretrained(
                config['model']['name'], torch_dtype=torch.bfloat16)
    
    # Load datasets
    train_dataset = load_from_disk(config['data']['train_path'])
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size = config['data']['train_batch_size'], 
        shuffle = True)
    
    val_dataset = load_from_disk(config['data']['val_path'])
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset, 
        batch_size = config['data']['eval_batch_size'], 
        shuffle = True)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['training']['learning_rate'])
    model, optimizer = fabric.setup(model, optimizer)
    train_dataloader, val_dataloader = fabric.setup_dataloaders(train_dataloader, val_dataloader)
    
    date = time.strftime("%Y-%m-%d_%H-%M-%S", time.gmtime())
    out_dir = os.path.join("models", config['training']['out_name'], date)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, 'config.yaml'), 'w') as f:
        yaml.dump(config, f)
    
    step_count = 0
    iter_num = 0
    min_loss = float("inf")
    model.train()
    for epoch in range(config['training']['num_epochs']):
        fabric.print(f"Epoch {epoch}")
        for batch in train_dataloader:
            if step_count <= config['training']['warmup_steps']:
                # linear warmup
                lr = config['training']['learning_rate'] * step_count / config['training']['warmup_steps']
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr
            
            t0 = time.time()
            batch = preprocess_batch(batch, fabric)
            model_out = model(**batch)
            loss = model_out.loss
            fabric.backward(loss)
            
            if (iter_num + 1) % config['training']['gradient_accumulation_steps'] == 0:
                optimizer.step()
                optimizer.zero_grad()
                step_count += 1
                
                if step_count % config['training']['save_interval'] == 0:
                    val_loss = validate(fabric, model, tokenizer, val_dataloader)
                    # scheduler.step(val_loss)
                    fabric.barrier()
                    
                    if fabric.global_rank == 0:
                        fabric.print("Validation loss: {:.4f}".format(val_loss))
                        if val_loss < min_loss:
                            fabric.print(f"Validation loss decreased from {min_loss:.4f} to {val_loss:.4f}")
                            min_loss = val_loss
                            fabric.print(f"Saving weights to {out_dir}")
                            best_dir = os.path.join(out_dir, 'best')
                            os.makedirs(best_dir, exist_ok=True)
                            checkpoint_path = os.path.join(best_dir, f"best-{iter_num:06d}-{val_loss:.4f}-ckpt.pth")
                            model.save_pretrained(checkpoint_path)
                        
                        iter_dir = os.path.join(out_dir, f"iters")
                        os.makedirs(iter_dir, exist_ok=True)
                        if len(os.listdir(iter_dir)) > config['training']['max_num_checkpoints']:
                            # Delete oldest folder
                            oldest = min(os.listdir(iter_dir))
                            shutil.rmtree(os.path.join(iter_dir, oldest))
                        checkpoint_path = os.path.join(iter_dir, f"iter-{iter_num:06d}-{val_loss:.4f}-ckpt.pth")
                        model.save_pretrained(checkpoint_path)
                    fabric.barrier()
            
            dt = time.time() - t0
            if iter_num % config['training']['log_interval'] == 0:
                fabric.print(f"iter {iter_num}: loss {loss.item():.4f}, time: {dt*1000:.2f}ms")
            iter_num += 1

@torch.inference_mode()
def validate(fabric: L.Fabric, 
             model: torch.nn.Module, 
             tokenizer: AutoTokenizer,
             val_dataloader: torch.utils.data.DataLoader)-> torch.Tensor:
    fabric.print("Validating ...")
    model.eval()
    data_iter = iter(val_dataloader)
    losses = torch.zeros(config['training']['eval_iters'])
    for k in range(config['training']['eval_iters']):
        try:
            batch = next(data_iter)
        except StopIteration:
            break
        
        batch = preprocess_batch(batch, fabric)
        model_out = model(**batch)
        loss = model_out.loss.mean()
        losses[k] = loss.item()
    out = losses.mean()
    
    if fabric.global_rank == 0:
        print("Generating samples ...")
        data_iter = iter(val_dataloader)
        batch = next(data_iter)
        
        batch = preprocess_batch(batch, fabric)
        batch['input_ids'] = batch['input_ids'][0].unsqueeze(0)
        batch['attention_mask'] = batch['attention_mask'][0].unsqueeze(0)
        batch['labels'] = batch['labels'][0].unsqueeze(0)
        
        outputs = model.generate(
            input_ids = batch['input_ids'],
            attention_mask = batch['attention_mask'],
            max_new_tokens = 64, 
            do_sample = True, 
            top_p = 0.9, 
            eos_token_id = tokenizer.eos_token_id,
            pad_token_id = tokenizer.eos_token_id)
        
        inputs = tokenizer.batch_decode(batch['input_ids'].detach().cpu().numpy(), skip_special_tokens=True)
        fabric.print("Model Input:", inputs)
        
        preds = tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True)
        fabric.print("Model Output: ", preds[0])
        
        batch['labels'][batch['labels'] == -100] = tokenizer.pad_token_id
        labels = tokenizer.batch_decode(batch['labels'].detach().cpu().numpy(), skip_special_tokens=True)
        fabric.print("Target: ", labels[0])
    
    model.train()
    return out.item()

if __name__ == "__main__":
    torch.set_float32_matmul_precision('medium')
    main()
