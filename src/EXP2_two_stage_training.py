import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import BertConfig, BertForMaskedLM, BertModel, BertTokenizerFast, get_linear_schedule_with_warmup
from datasets import load_dataset
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import random
import gc 

from electra import ElectraModel
from electra import set_seed, SEQ_LEN, BATCH_SIZE, LR, DEVICE
from EXP1_train_wikitext import get_dataloader

TOTAL_STEPS = 1000     

set_seed(42)

def run_experiment(hidden_size, dataloader, training_style="two_stage"):
    print(f"\n=== Training: Size {hidden_size} | Style: {training_style} ===")
    
    tie_weights = True if training_style == "joint" else False
    model = ElectraModel(hidden_size, tie_weights=tie_weights).to(DEVICE)
    losses = []
    
    data_iter = iter(dataloader)

    if training_style == "joint":
        optimizer = torch.optim.AdamW(list(model.generator.parameters()) + list(model.discriminator.parameters()), lr=LR)
        scheduler = get_linear_schedule_with_warmup(optimizer, 100, TOTAL_STEPS)
        model.train()

        for _ in tqdm(range(TOTAL_STEPS), desc=f"Joint - Size {hidden_size}"):
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader)
                batch = next(data_iter)
            
            optimizer.zero_grad()
            total_loss, disc_loss_val = model(batch["input_ids"].to(DEVICE), batch["attention_mask"].to(DEVICE), mode='joint')
            total_loss.backward()
            optimizer.step()
            scheduler.step()
            losses.append(disc_loss_val.item())

    elif training_style in ["two_stage", "two_stage_no_transfer"]:
        mid_point = TOTAL_STEPS//2
        
        optimizer = torch.optim.AdamW(model.generator.parameters(), lr=LR)
        scheduler = get_linear_schedule_with_warmup(optimizer, 100, mid_point)
        model.train()
        
        for _ in tqdm(range(mid_point), desc=f"Stg 1 (Gen) - Size {hidden_size}"):
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader)
                batch = next(data_iter)
            
            optimizer.zero_grad()
            loss = model(batch["input_ids"].to(DEVICE), batch["attention_mask"].to(DEVICE), mode='gen_only')
            loss.backward()
            optimizer.step()
            scheduler.step()
            losses.append(0.693) 

        if training_style == "two_stage":
            model.transfer_weights()
            
        model.freeze_generator()

        optimizer = torch.optim.AdamW(model.discriminator.parameters(), lr=LR)
        scheduler = get_linear_schedule_with_warmup(optimizer, 100, mid_point)

        for _ in tqdm(range(mid_point), desc=f"Stg 2 (Disc) - Size {hidden_size}"):
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader)
                batch = next(data_iter)
            
            optimizer.zero_grad()
            loss = model(batch["input_ids"].to(DEVICE), batch["attention_mask"].to(DEVICE), mode='disc_only')
            loss.backward()
            optimizer.step()
            scheduler.step()
            losses.append(loss.item())

    del model
    del optimizer
    torch.cuda.empty_cache()
    gc.collect()
    
    return losses

if __name__ == "__main__":
    loader = get_dataloader()
    
    sizes = [64, 128] 
    results_two_stage = {}
    results_joint = {}
    results_no_transfer = {}
    
    for size in sizes:
        results_two_stage[size] = run_experiment(size, loader, training_style="two_stage")

    for size in sizes:
        results_joint[size] = run_experiment(size, loader, training_style="joint")
        
    for size in sizes:
        results_no_transfer[size] = run_experiment(size, loader, training_style="two_stage_no_transfer")

    plt.figure(figsize=(8, 6))
    
    def smooth(y, box_pts=40):
        box = np.ones(box_pts)/box_pts
        y_smooth = np.convolve(y, box, mode='valid')
        return y_smooth

    colors = {128: '#ff7f0e', 64: '#2ca02c'}
    
    for size, loss_data in results_two_stage.items():
        smoothed = smooth(loss_data)
        plt.plot(smoothed, label=f"Two-Stage (Transfer) {size}", color=colors[size], linewidth=2, linestyle='-')

    for size, loss_data in results_joint.items():
        smoothed = smooth(loss_data)
        plt.plot(smoothed, label=f"Joint Training {size}", color=colors[size], linewidth=2, linestyle='--')
        
    for size, loss_data in results_no_transfer.items():
        smoothed = smooth(loss_data)
        plt.plot(smoothed, label=f"Two-Stage (No Transfer) {size}", color=colors[size], linewidth=2, linestyle=':')

    break_point = TOTAL_STEPS//2
    plt.axvline(x=break_point, color='gray', linestyle='-.', linewidth=1, label="Stage Transition Point")
    
    plt.title("Comparison: Joint vs Two-Stage (Transfer vs No Transfer)", fontsize=16)
    plt.xlabel("Training Steps", fontsize=12)
    plt.ylabel("Discriminator Loss (Smoothed)", fontsize=12)
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("imgs/experiment_comparison_all_6_models.png", dpi=300)
    plt.show()