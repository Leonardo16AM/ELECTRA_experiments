import torch
import torch.nn as nn
from transformers import BertConfig, BertModel, BertForMaskedLM
from .electra import set_seed,SEQ_LEN,VOCAB_SIZE,DEVICE
set_seed(42)

class ElectraModel(nn.Module):
    def __init__(self, hidden_size, tie_weights=False):
        super().__init__()
        self.config=BertConfig(
            vocab_size=VOCAB_SIZE,
            hidden_size=hidden_size,
            num_hidden_layers=4,        
            num_attention_heads=4,
            intermediate_size=hidden_size * 4,
            max_position_embeddings=SEQ_LEN
        )
        
        self.generator=BertForMaskedLM(self.config)
        self.discriminator=BertModel(self.config)
        self.disc_head=nn.Linear(hidden_size, 1)

        if tie_weights:
            self.discriminator.embeddings=self.generator.bert.embeddings
            self.discriminator.embeddings.word_embeddings=self.generator.bert.embeddings.word_embeddings
            self.discriminator.embeddings.position_embeddings=self.generator.bert.embeddings.position_embeddings
            self.discriminator.embeddings.token_type_embeddings=self.generator.bert.embeddings.token_type_embeddings

    def forward(self, input_ids, attention_mask, mode='joint'):
        probability_matrix=torch.full(input_ids.shape, 0.15).to(DEVICE)
        masked_indices=torch.bernoulli(probability_matrix).bool()
        masked_indices=masked_indices & (input_ids != 101) & (input_ids != 102) & (input_ids != 0)
        
        gen_labels=input_ids.clone()
        gen_labels[~masked_indices]=-100 
        masked_input_ids=input_ids.clone()
        masked_input_ids[masked_indices]=103

        with torch.set_grad_enabled(mode != 'disc_only'):
            gen_outputs=self.generator(masked_input_ids, attention_mask=attention_mask, labels=gen_labels)
            gen_loss=gen_outputs.loss
            gen_logits=gen_outputs.logits

        if mode == 'gen_only':
            return gen_loss

        with torch.no_grad():
            masked_logits=gen_logits[masked_indices]
            probs=torch.softmax(masked_logits, dim=-1)
            sampled_tokens=torch.multinomial(probs, 1).squeeze()
            corrupted_input_ids=input_ids.clone()
            corrupted_input_ids[masked_indices]=sampled_tokens
            disc_labels=(corrupted_input_ids != input_ids).float()

        disc_outputs=self.discriminator(corrupted_input_ids, attention_mask=attention_mask)
        disc_logits=self.disc_head(disc_outputs.last_hidden_state).squeeze(-1)

        bce_loss=nn.BCEWithLogitsLoss()
        active_loss=attention_mask.view(-1) == 1
        disc_loss=bce_loss(disc_logits.view(-1)[active_loss], disc_labels.view(-1)[active_loss])
        
        if mode == 'disc_only':
            return disc_loss
        
        if mode == 'joint':
            total_loss=gen_loss + 50.0 * disc_loss
            return total_loss, disc_loss

    def transfer_weights(self):
        self.discriminator.load_state_dict(self.generator.bert.state_dict(), strict=False)

    def freeze_generator(self):
        for param in self.generator.parameters():
            param.requires_grad=False
