import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from transformers import AutoTokenizer
import torch.optim as optim


# tokenizer
tokenizer = AutoTokenizer.from_pretrained("gpt2")


with open("data.txt", "r", encoding="utf-8") as f:
    text = f.read()

# tokens
encodings = tokenizer(text, return_tensors="pt")
data = encodings["input_ids"][0]

# batch
def get_batch(data, batch_size=16, block_size=64):
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x, y

device = "cuda" if torch.cuda.is_available() else "cpu"



class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, n_head):
        super().__init__()
        assert d_model % n_head == 0
        
        self.n_head = n_head
        self.head_dim = d_model // n_head
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, x, mask=None):
        B, T, C = x.size()
        qkv = self.qkv(x)                    # (B, T, 3*C)
        q, k, v = qkv.chunk(3, dim=-1)       # (B, T, C)

        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)  # (B, nh, T, hd)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)    # (B, nh, T, T)
        if mask is not None:
            att = att.masked_fill(mask == 0, float('-inf'))
        att = F.softmax(att, dim=-1)

        out = att @ v                       # (B, nh, T, hd)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.out(out)

class FeedForward(nn.Module):
    def __init__(self, d_model, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, d_model),
        )
    def forward(self, x):
        return self.net(x)

#  Transformer Block 
class Block(nn.Module):
    def __init__(self, d_model, n_head, ff_hidden_dim):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.attn = MultiHeadSelfAttention(d_model, n_head)
        self.ff = FeedForward(d_model, ff_hidden_dim)

    def forward(self, x, mask=None):
        x = x + self.attn(self.ln1(x), mask)
        x = x + self.ff(self.ln2(x))
        return x


class MiniGPT(nn.Module):
    def __init__(self, vocab_size, d_model=512, n_layer=8, n_head=8, block_size=64):
        super().__init__()
        self.block_size = block_size
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(block_size, d_model)
        
        self.blocks = nn.ModuleList(*[
            Block(d_model, n_head, 4*d_model)
            for _ in range(n_layer)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, idx, targets=None):
        B, T = idx.size()
        assert T <= self.block_size

        tok_emb = self.token_emb(idx)                     # (B, T, d_model)
        pos_emb = self.pos_emb(torch.arange(T, device=idx.device))  # (T, d_model)
        x = tok_emb + pos_emb                             # (B, T, d_model)

        # causal mask 不看未來的 token
        mask = torch.tril(torch.ones(T, T, device=idx.device)).unsqueeze(0).unsqueeze(0)
        x = self.blocks(x, mask)
        x = self.ln_f(x)
        logits = self.head(x)                             # (B, T, vocab_size)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss


# 5. training
vocab_size = tokenizer.vocab_size
model = MiniGPT(vocab_size, d_model=512, n_layer=8, n_head=8, block_size=64)
optimizer = optim.AdamW(model.parameters(), lr=3e-4)

for step in range(1000):
    x, y = get_batch(data)
    logits, loss = model(x, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step % 100 == 0:
        print(f"Step {step}: loss = {loss.item():.4f}")


