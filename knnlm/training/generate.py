import argparse
import os

from datasets import load_dataset
from transformers import AutoTokenizer

# Define and parse command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type=str, default="t5-large")
parser.add_argument('--data_path', type=str, default="data/data.jsonl")
args = parser.parse_args()

def tokenize_text(batch, tokenizer, max_length=128):
    prompts = batch['input']
    responses = batch['output']
    
    prompt_encodings = tokenizer.batch_encode_plus(
        prompts, 
        padding='max_length',
        truncation=True,
        max_length=max_length,
        return_tensors='pt')
    
    response_encodings = tokenizer.batch_encode_plus(
        responses,
        padding='max_length',
        truncation=True,
        max_length=max_length,
        return_tensors='pt')
    
    response_encodings['input_ids'][response_encodings['input_ids'] == 0] = -100 
    
    model_inputs = {
        'input_ids': prompt_encodings['input_ids'],
        'attention_mask': prompt_encodings['attention_mask'],
        'labels': response_encodings['input_ids']
    }
    return model_inputs

def main():
    # Load pretrained model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        
    # Load datasets
    dataset = load_dataset("json", data_files=args.data_path, split="train")
    dataset = dataset.map(
        lambda batch: tokenize_text(batch, tokenizer),
        remove_columns = dataset.column_names,
        batched = True,
        num_proc = 4)
    
    input_path = os.path.split(args.data_path)
    save_path = os.path.join(
        input_path[0], 'tokenized',
        args.model_name,
        input_path[-1].replace('.jsonl', '.arrow'))
    
    dataset.save_to_disk(save_path)

if __name__ == "__main__":
    main()
