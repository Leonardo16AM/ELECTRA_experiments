import torch
import torch.nn as nn
from transformers import BertConfig, BertModel, BertForMaskedLM
import random
import numpy as np
import os

# Configuration based on ELECTRA-Small
SEQ_LEN = 128           
BATCH_SIZE = 32         
VOCAB_SIZE = 30522 
TRAIN_STEPS = 1000      
LR = 5e-4               

DISC_HIDDEN_SIZE = 256  
GEN_HIDDEN_SIZE = 64 
NUM_LAYERS = 12        

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ElectraGenerator(nn.Module):
    def __init__(self, config, embedding_size):
        super().__init__()
        self.config = config
        self.bert = BertForMaskedLM(config)
        
        self.embedding_projection = nn.Linear(embedding_size, config.hidden_size)

    def forward(self, inputs_embeds, attention_mask, labels=None):
        hidden_states = self.embedding_projection(inputs_embeds)
        
        outputs = self.bert(inputs_embeds=hidden_states, 
                            attention_mask=attention_mask, 
                            labels=labels)
        return outputs

class SimpleELECTRA(nn.Module):
    def __init__(self, disc_hidden_size=256, gen_hidden_size=64):
        super().__init__()
        
        self.disc_config = BertConfig(
            vocab_size=VOCAB_SIZE,
            hidden_size=disc_hidden_size,
            num_hidden_layers=NUM_LAYERS,      
            num_attention_heads=4,             
            intermediate_size=disc_hidden_size * 4,
            max_position_embeddings=SEQ_LEN
        )
        self.discriminator = BertModel(self.disc_config)
        self.disc_head = nn.Linear(disc_hidden_size, 1)

        self.gen_config = BertConfig(
            vocab_size=VOCAB_SIZE,
            hidden_size=gen_hidden_size,       
            num_hidden_layers=NUM_LAYERS,      
            num_attention_heads=1,         
            intermediate_size=gen_hidden_size * 4,
            max_position_embeddings=SEQ_LEN
        )
        self.generator = ElectraGenerator(self.gen_config, embedding_size=disc_hidden_size)

    def forward(self, input_ids, attention_mask):
        probability_matrix = torch.full(input_ids.shape, 0.15).to(input_ids.device)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        masked_indices = masked_indices & (input_ids != 101) & (input_ids != 102) & (input_ids != 0)
        
        gen_labels = input_ids.clone()
        gen_labels[~masked_indices] = -100  

        masked_input_ids = input_ids.clone()
        
        indices_replaced = torch.bernoulli(torch.full(input_ids.shape, 0.8)).bool().to(device) & masked_indices
        masked_input_ids[indices_replaced] = 103

        indices_random = torch.bernoulli(torch.full(input_ids.shape, 0.5)).bool().to(device) & masked_indices & ~indices_replaced
        random_words = torch.randint(VOCAB_SIZE, input_ids.shape, dtype=torch.long).to(device)
        masked_input_ids[indices_random] = random_words[indices_random]

        shared_embeddings = self.discriminator.embeddings(masked_input_ids)

        gen_outputs = self.generator(inputs_embeds=shared_embeddings, 
                                     attention_mask=attention_mask, 
                                     labels=gen_labels)
        gen_loss = gen_outputs.loss
        gen_logits = gen_outputs.logits

        with torch.no_grad():
            masked_logits = gen_logits[masked_indices]
            probs = torch.softmax(masked_logits, dim=-1)
            sampled_tokens = torch.multinomial(probs, 1).squeeze()
            
            corrupted_input_ids = input_ids.clone()
            corrupted_input_ids[masked_indices] = sampled_tokens
            
            disc_labels = (corrupted_input_ids != input_ids).float()

        disc_outputs = self.discriminator(corrupted_input_ids, attention_mask=attention_mask)
        hidden_states = disc_outputs.last_hidden_state
        disc_logits = self.disc_head(hidden_states).squeeze(-1)

        bce_loss = nn.BCEWithLogitsLoss()
        
        active_loss = attention_mask.view(-1) == 1
        active_logits = disc_logits.view(-1)[active_loss]
        active_labels = disc_labels.view(-1)[active_loss]
        
        disc_loss = bce_loss(active_logits, active_labels)

        total_loss = gen_loss + 50.0 * disc_loss
        
        return total_loss, disc_loss.item(), gen_loss.item()

    def save_checkpoint(self, path):
        torch.save(self.state_dict(), path)

    @classmethod
    def load_checkpoint(cls, path, disc_hidden=256, gen_hidden=64, device='cpu'):
        model = cls(disc_hidden_size=disc_hidden, gen_hidden_size=gen_hidden)
        model.load_state_dict(torch.load(path, map_location=device))
        model.to(device)
        return model
