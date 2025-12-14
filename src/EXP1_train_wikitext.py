import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import BertConfig, BertForMaskedLM, BertModel, BertTokenizerFast
from datasets import load_dataset
import matplotlib.pyplot as plt
import random
import numpy as np
from tqdm import tqdm

from electra import *

set_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_dataloader():
    print("Loading WikiText-2 dataset (small version)...")
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    dataset = dataset.select(range(4000)) 
    
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

    def encode(examples):
        return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=SEQ_LEN)

    dataset = dataset.filter(lambda x: len(x["text"])>10) 
    encoded_dataset = dataset.map(encode, batched=True)
    encoded_dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])
    
    return DataLoader(encoded_dataset, batch_size=BATCH_SIZE, shuffle=True)

def train_experiment(gen_hidden_size, dataloader):
    model = SimpleELECTRA(gen_hidden_size=gen_hidden_size, disc_hidden_size=256).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    
    disc_losses = []
    model.train()
    
    progress_bar = tqdm(range(TRAIN_STEPS))
    data_iter = iter(dataloader)
    
    for step in progress_bar:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)
            
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        
        optimizer.zero_grad()
        loss, d_loss, g_loss = model(input_ids, attention_mask)
        loss.backward()
        optimizer.step()
        
        disc_losses.append(d_loss)
        if step % 50 == 0:
            progress_bar.set_description(f"D_Loss: {d_loss:.4f} | G_Loss: {g_loss:.4f}")

    model.save_checkpoint(f"models/electra_gen{gen_hidden_size}_disc256.pth")
    return disc_losses









if __name__ == "__main__":
    train_loader = get_dataloader()
    
    gen_sizes = [64, 128, 256,528] 
    results = {}

    for size in gen_sizes:
        loss_history = train_experiment(size, train_loader)
        smoothed_loss = np.convolve(loss_history, np.ones(20)/20, mode='valid')
        results[size] = smoothed_loss

    plt.figure(figsize=(10, 6))
    for size, losses in results.items():
        plt.plot(losses, label=f"Generator Size {size}")
    
    plt.title("Effect of Generator Size on Discriminator Training")
    plt.xlabel("Training Steps")
    plt.ylabel("Discriminator Loss (Smoothed)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    output_file = "imgs/electra_experiment_1_en.png"
    plt.savefig(output_file)
    plt.show()