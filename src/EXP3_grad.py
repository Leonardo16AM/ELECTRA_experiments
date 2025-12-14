from __future__ import annotations

import random
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from datasets import load_dataset
from transformers import BertTokenizerFast, BertConfig, BertForMaskedLM

from electra import SimpleELECTRA, set_seed, SEQ_LEN, VOCAB_SIZE


def pick_out_dir():
    here = Path(__file__).resolve().parent
    candidates = [here / "imgs", here.parent / "imgs", here / ".." / "imgs"]
    for c in candidates:
        c = c.resolve()
        if c.exists():
            return c
    out = (here.parent / "imgs").resolve()
    out.mkdir(parents=True, exist_ok=True)
    return out


def get_one_batch(tokenizer: BertTokenizerFast, batch_size: int = 32, n_rows: int = 4000):
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    ds = ds.filter(lambda x: x["text"] is not None and len(x["text"]) > 10)
    ds = ds.select(range(min(n_rows, len(ds))))

    def enc(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=SEQ_LEN,
        )

    ds = ds.map(enc, batched=True, remove_columns=["text"])
    ds.set_format(type="torch", columns=["input_ids", "attention_mask"])

    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)
    return next(iter(loader))


def special_mask(tokenizer: BertTokenizerFast, input_ids: torch.Tensor) -> torch.Tensor:
    # input_ids: [B, S]
    masks = []
    for row in input_ids.tolist():
        masks.append(tokenizer.get_special_tokens_mask(row, already_has_special_tokens=True))
    return torch.tensor(masks, dtype=torch.bool)


def make_mlm_model(hidden: int = 256, layers: int = 12, heads: int = 4) -> BertForMaskedLM:
    cfg = BertConfig(
        vocab_size=VOCAB_SIZE,
        hidden_size=hidden,
        num_hidden_layers=layers,
        num_attention_heads=heads,
        intermediate_size=hidden * 4,
        max_position_embeddings=SEQ_LEN,
    )
    return BertForMaskedLM(cfg)


def mask_for_mlm(
    tokenizer: BertTokenizerFast,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    p: float = 0.15,
):
    x = input_ids.clone()
    y = input_ids.clone()

    spec = special_mask(tokenizer, input_ids)
    prob = torch.full_like(input_ids, p, dtype=torch.float)
    prob.masked_fill_(spec, 0.0)
    prob.masked_fill_(attention_mask == 0, 0.0)

    masked = torch.bernoulli(prob).bool()
    y[~masked] = -100

    # 80% -> [MASK]
    replace = (torch.rand_like(prob) < 0.8) & masked
    x[replace] = tokenizer.mask_token_id

    # 10% -> random token (of the remaining masked)
    rand = (torch.rand_like(prob) < 0.5) & masked & ~replace
    random_words = torch.randint(VOCAB_SIZE, x.shape, dtype=torch.long)
    x[rand] = random_words[rand]

    # remaining 10% keep as-is
    return x, y, masked, (~spec) & (attention_mask == 1)


def plot_heatmap(G: torch.Tensor, dots: torch.Tensor | None, title: str, save_path: Path):
    G = G.detach().cpu().numpy()

    plt.figure(figsize=(12, 4))
    plt.imshow(G, aspect="auto")
    plt.title(title)
    plt.xlabel("Token position")
    plt.ylabel("Example in batch")
    plt.colorbar(label=r"$\left\|\partial \mathcal{L} / \partial H\right\|_2$")

    if dots is not None:
        M = dots.detach().cpu().numpy().astype(bool)
        ys, xs = np.where(M)
        plt.scatter(xs, ys, s=6)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"saved: {save_path}")


def main():
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = pick_out_dir()

    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
    batch = get_one_batch(tokenizer, batch_size=32, n_rows=4000)

    # keep the heatmaps readable
    K = min(16, batch["input_ids"].shape[0])
    input_ids = batch["input_ids"][:K].to(device)
    attn = batch["attention_mask"][:K].to(device)

    # -------- MLM heatmap --------
    mlm = make_mlm_model(hidden=256, layers=12, heads=4).to(device)
    mlm.eval()

    masked_ids, mlm_labels, masked_pos, valid_pos = mask_for_mlm(
        tokenizer, input_ids, attn, p=0.15
    )

    mlm.zero_grad(set_to_none=True)
    out = mlm(
        input_ids=masked_ids,
        attention_mask=attn,
        labels=mlm_labels,
        output_hidden_states=True,
        return_dict=True,
    )
    H = out.hidden_states[-1]
    H.retain_grad()

    out.loss.backward()
    G_mlm = H.grad.norm(dim=-1)

    plot_heatmap(
        G_mlm,
        dots=masked_pos,
        title="MLM: gradient per position (dots = masked tokens)",
        save_path=out_dir / "heatmap_mlm.png",
    )

    # RTD heatmap
    electra = SimpleELECTRA(disc_hidden_size=256, gen_hidden_size=64).to(device)
    electra.eval()

    # use the same masking positions to build the corrupted sequence
    masked_ids, _, masked_pos, valid_disc = mask_for_mlm(
        tokenizer, input_ids, attn, p=0.15
    )

    with torch.no_grad():
        shared_emb = electra.discriminator.embeddings(masked_ids)
        gen_out = electra.generator(inputs_embeds=shared_emb, attention_mask=attn, labels=None)
        gen_logits = gen_out.logits  # [B,S,V]

        masked_logits = gen_logits[masked_pos]  # [Nmask,V]
        probs = torch.softmax(masked_logits, dim=-1)
        sampled = torch.multinomial(probs, 1).squeeze(-1)

        corrupted = input_ids.clone()
        corrupted[masked_pos] = sampled

        disc_labels = (corrupted != input_ids).float()
        replaced_pos = (disc_labels > 0.5) & valid_disc

    electra.zero_grad(set_to_none=True)
    disc_out = electra.discriminator(corrupted, attention_mask=attn)
    Hd = disc_out.last_hidden_state
    Hd.retain_grad()

    disc_logits = electra.disc_head(Hd).squeeze(-1)
    bce = nn.functional.binary_cross_entropy_with_logits(disc_logits, disc_labels, reduction="none")
    loss_rtd = (bce * valid_disc.float()).sum() / (valid_disc.float().sum() + 1e-8)

    loss_rtd.backward()
    G_rtd = Hd.grad.norm(dim=-1)

    plot_heatmap(
        G_rtd,
        dots=replaced_pos,
        title="RTD (discriminator): gradient per position (dots = replaced tokens)",
        save_path=out_dir / "heatmap_rtd.png",
    )

    print("done.")


if __name__ == "__main__":
    main()
