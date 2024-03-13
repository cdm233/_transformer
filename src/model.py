from safetensors.torch import save_model, load_model
import torch
import torch.nn as nn
import shutil
from torch.nn import functional as F
from tqdm import tqdm
import os
import math
import bitsandbytes as bnb

dropout_rate = 0
block_size = 0
model_version = 0
feed_forward_multiplier = 0
model_initialized = False
dataloader = None
model_arch = {}


def initialize_model(conf, loader):
    global dropout_rate
    global block_size
    global model_initialized
    global dataloader
    global model_version
    global feed_forward_multiplier
    global model_arch

    dataloader = loader
    feed_forward_multiplier = conf["feed_forward_multiplier"]
    model_version = conf["model_version"]
    dropout_rate = conf["dropout_rate"]
    block_size = conf["block_size"]

    model_arch = conf["model_arch"]

    model_initialized = True


class AttentionHead(nn.Module):
    def __init__(self, head_size, embed_dim):
        super(AttentionHead, self).__init__()
        self.head_size = head_size

        self.key = nn.Linear(embed_dim, self.head_size, bias=False)
        self.query = nn.Linear(embed_dim, self.head_size, bias=False)
        self.value = nn.Linear(embed_dim, self.head_size, bias=False)

        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        B, T, C = x.shape

        k = self.key(x)
        q = self.query(x)
        v = self.value(x)

        wei = q @ k.transpose(-2, -1) * C ** -0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)

        out = wei @ v

        return out


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, head_size):
        super(MultiHeadAttention, self).__init__()
        self.heads = nn.ModuleList([AttentionHead(head_size, embed_dim=embed_dim) for _ in range(num_heads)])
        self.proj = nn.Linear(embed_dim, embed_dim)   # linear projection back
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = torch.cat([head(x) for head in self.heads], dim=-1)
        x = self.proj(x)
        return self.dropout(x)


class SwiGLU(nn.Module):
    def forward(self, x):
        x, gate = x.chunk(2, dim=-1)
        return F.silu(gate) * x


class FeedForward(nn.Module):
    def __init__(self, n_embd, out_n_embd=None):
        super(FeedForward, self).__init__()
        if out_n_embd is None or out_n_embd == 0:
            out_n_embd = n_embd

        self.layer = nn.Sequential(
            nn.Linear(n_embd, n_embd * feed_forward_multiplier),
            # nn.LeakyReLU(),
            SwiGLU(),
            nn.Linear(n_embd * feed_forward_multiplier // 2, out_n_embd),
            nn.Dropout(dropout_rate)
        )

    def forward(self, x):
        return self.layer(x)


class RMSNorm(nn.Module):
    def __init__(self, d, p=-1., eps=1e-8, bias=False):
        super(RMSNorm, self).__init__()

        self.eps = eps
        self.d = d
        self.p = p
        self.bias = bias

        self.scale = nn.Parameter(torch.ones(d))
        self.register_parameter("scale", self.scale)

        if self.bias:
            self.offset = nn.Parameter(torch.zeros(d))
            self.register_parameter("offset", self.offset)

    def forward(self, x):
        if self.p < 0. or self.p > 1.:
            norm_x = x.norm(2, dim=-1, keepdim=True)
            d_x = self.d
        else:
            partial_size = int(self.d * self.p)
            partial_x, _ = torch.split(x, [partial_size, self.d - partial_size], dim=-1)

            norm_x = partial_x.norm(2, dim=-1, keepdim=True)
            d_x = partial_size

        rms_x = norm_x * d_x ** (-1. / 2)
        x_normed = x / (rms_x + self.eps)

        if self.bias:
            return self.scale * x_normed + self.offset

        return self.scale * x_normed


class AttentionBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_out_n_embd=None):
        super(AttentionBlock, self).__init__()
        head_size = embed_dim // num_heads
        self.MHA = MultiHeadAttention(head_size=head_size, embed_dim=embed_dim, num_heads=num_heads)
        self.FF = FeedForward(embed_dim, out_n_embd=ff_out_n_embd)

        self.FF_out_n_embd = ff_out_n_embd

        # self.layer_norm1 = nn.LayerNorm(embed_dim)
        # self.layer_norm2 = nn.LayerNorm(embed_dim)

        self.layer_norm1 = RMSNorm(embed_dim)
        self.layer_norm2 = RMSNorm(embed_dim)

    def forward(self, x):
        x = x + self.MHA(self.layer_norm1(x))

        x = x + self.FF(self.layer_norm2(x))

        return x

    
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        return x + self.pe[:x.size(0), :]


class NoamLR:
    def __init__(self, optimizer, model_size, warmup_steps, factor=1):
        self.optimizer = optimizer
        self.model_size = model_size
        self.warmup_steps = warmup_steps
        self.factor = factor
        self._step = 0

    def step(self):
        self._step += 1
        lr = self.factor * (self.model_size ** (-0.5) *
                            min(self._step ** (-0.5), self._step * self.warmup_steps ** (-1.5)))
        for p in self.optimizer.param_groups:
            p['lr'] = lr


class TransformerNet(nn.Module):
    def __init__(self, vocab_size):
        super(TransformerNet, self).__init__()
        self.model_param = {
            "dropout_rate": dropout_rate,
            "block_size": block_size,
            "model_arch": model_arch,
            "model_initialized": model_initialized,
        }
        assert model_initialized, "Model not initialized!"

        self.model_arch = model_arch

        self.model_version = model_version

        self.temperature = 0.75

        assert self.validate_model_arch(), "The model architecture defined is faulty!"

        embedding_arch = self.model_arch["embedding"]
        block_arch = self.model_arch["block"]
        lm_head_arch = self.model_arch["lm_head"]

        self.emb_dim = embedding_arch["embed_dim"]

        self.pos_encoder = PositionalEncoding(d_model=embedding_arch["embed_dim"], max_len=block_size)

        # self.token_embedding = nn.Embedding(vocab_size, embedding_arch["embed_dim"])
        self.token_embedding = bnb.nn.StableEmbedding(vocab_size, embedding_arch["embed_dim"])
        self.lm_head = nn.Sequential()

        last_lm_layer = block_arch["out_embed_dim"]
        for lm_layer_arch in lm_head_arch:
            self.lm_head.append(nn.Linear(last_lm_layer, lm_layer_arch))
            # self.lm_head.append(SwiGLU())
            self.lm_head.append(nn.LeakyReLU())
            self.lm_head.append(nn.Dropout(dropout_rate))
            last_lm_layer = lm_layer_arch

        self.lm_head.append(nn.Linear(last_lm_layer, vocab_size))

        self.attention_blocks = nn.ModuleList()

        cur_rep = block_arch["num_rep"]
        cur_num_heads = block_arch["num_heads"]
        cur_embed_dim = block_arch["embed_dim"]

        for i in range(cur_rep):
            if i == cur_rep - 1:
                FF_out_n_embd = block_arch.get("out_embed_dim", None)
            else:
                FF_out_n_embd = None

            self.attention_blocks.append(AttentionBlock(cur_embed_dim, cur_num_heads, ff_out_n_embd=FF_out_n_embd))

        # self.layer_norm = nn.LayerNorm(block_arch[-1]["out_embed_dim"])
        self.layer_norm = RMSNorm(block_arch["out_embed_dim"])

        parent_dir = f"model_v{self.model_version}"
        full_path = f"./models/{parent_dir}"

        if not os.path.exists(full_path):
            os.mkdir(full_path)

        shutil.copyfile("./config.json", f"{full_path}/config.json")       

    def convert_model_to_fp16(model):
        # Convert the entire model to fp16
        model.half()
        
        # Convert specific layers back to fp32
        for name, module in model.named_modules():
            if isinstance(module, (torch.nn.Embedding, torch.nn.LayerNorm)):
                module.float()

    def convert_model(self, load_mode):
        self.load_mode = load_mode

        if load_mode == 'bfp16':
            load_dtype = torch.bfloat16
        elif load_mode == 'fp16':
            load_dtype = torch.float16
        else:
            load_dtype = torch.float16
            
        # Convert the entire model to fp16
        self = self.to(load_dtype)
        
        # Convert specific layers back to fp32
        for name, module in self.named_modules():
            if isinstance(module, (torch.nn.Embedding, torch.nn.LayerNorm)):
                module.float()

        self.load_dtype = load_dtype

    def forward(self, inputs, targets=None):
        tok_emb = self.token_embedding(inputs) * math.sqrt(self.emb_dim)
        x = self.pos_encoder(tok_emb)

        x = x.to(self.load_dtype)

        for block in self.attention_blocks:
            x = block(x)

        x = self.layer_norm(x)

        logits = self.lm_head(x)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens, end_token=None, mode=0):
        self.eval()
        if self.temperature < 0:
            self.temperature = 0

        if self.temperature > 1:
            self.temperature = 1

        for _ in tqdm(range(max_new_tokens), desc="Generating tokens"):
            logits, loss = self(idx[:, -self.model_param["block_size"]:])
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)

            target_idx = int(probs.shape[1] * self.temperature)

            # sorted_probs = sorted(probs.clone()[0], reverse=True)
            sorted_probs = torch.sort(probs.clone()[0], descending=True)[0]

            target = sorted_probs[target_idx]

            probs_mat = probs >= target
            probs_mat = probs_mat.to(torch.long)
            probs *= probs_mat

            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)

            if end_token != -1 and idx_next == end_token:
                print("End token hit, stopping...")
                break
        
        self.train()
        return idx

    def get_model_params(self):
        return sum(p.numel() for p in self.parameters())

    def save_model(self, name, estimate_loss=False):
        parent_dir = f"model_v{self.model_version}"

        if estimate_loss:
            loss = self.estimate_loss()
            save_model(self, f"./models/{parent_dir}/{name}_val_loss={loss}.safetensors")
        else:
            save_model(self, f"./models/{parent_dir}/{name}.safetensors")

    def load_model(self, name):
        parent_dir = f"model_v{self.model_version}"

        if "*" in name:
            for file_name in os.listdir(f"./models/{parent_dir}"):
                if name[:-1] in file_name:
                    load_model(self, f"./models/{parent_dir}/{file_name}")
                    break
        else:
            load_model(self, f"./models/{parent_dir}/{name}.safetensors")

    @torch.no_grad()
    def estimate_loss(self):
        eval_iterations = 10
        self.eval()

        losses = torch.zeros(eval_iterations)
        for k in range(eval_iterations):
            X, Y = dataloader.get_batch(batch_size=32, block_size=self.model_param["block_size"], train=False)
            logits, loss = self(X, Y)
            losses[k] = loss.item()

        out = losses.mean()
        self.train()

        out = "%.5f" % out
        return out

    def validate_model_arch(self):
        embedding_arch = self.model_arch["embedding"]
        block_arch = self.model_arch["block"]

        input_embedding_dim = embedding_arch["embed_dim"]
        input_block_embd = block_arch["embed_dim"]

        if input_embedding_dim != input_block_embd:
            return False

        return True


class Generator:
    def __init__(self, model, model_block_size, tokenizer, temperature=1, k=1000, p=0.5):
        self.model = model.eval()
        self.model_block_size = model_block_size
        self.tokenizer = tokenizer

        self.top_p_val = p
        self.top_k_val = k

        assert temperature > 0, "Temp must be greater than 0"

        self.temperature = temperature

    def format_output(self, tokenized_text):
        self.model.train()

        return self.tokenizer.Decode(tokenized_text[0].tolist()).replace("<s>", "").replace("</s>", " \n").replace('<n>', '\n')

    def random_sample(self, starting_text, max_new_tokens, end_token=None):
        for _ in tqdm(range(max_new_tokens), desc="Generating tokens"):
            logits, loss = self.model(starting_text[:, -self.model_block_size:])
            logits = logits[:, -1, :]

            logits, sorted_idx = torch.sort(logits, descending=True)

            top_k_logits = logits[0][:self.top_k_val]
            top_k_idx = sorted_idx[0][:self.top_k_val]

            probs = F.softmax(top_k_logits / self.temperature, dim=-1)

            idx_next_pos = torch.multinomial(probs, num_samples=1)
            starting_text = torch.cat((starting_text, top_k_idx[idx_next_pos].unsqueeze(dim=0)), dim=1)

            if end_token != -1 and top_k_idx[idx_next_pos] == end_token:
                print("End token hit, stopping...")
                break

        return self.format_output(starting_text)

    def top_p(self, starting_text, max_new_tokens, end_token=None):
        for _ in tqdm(range(max_new_tokens), desc="Generating tokens"):
            logits, loss = self.model(starting_text[:, -self.model_block_size:])
            logits = logits[:, -1, :]

            logits, sorted_idx = torch.sort(logits, descending=True)
            probs = F.softmax(logits / self.temperature, dim=-1)

            # probs.shape = [1, vocab_size]
            prob_sums = torch.cumsum(probs, dim=1)

            mask = prob_sums - probs > self.top_p_val
            probs[mask] = 0.0
            probs.div_(probs.sum(dim=-1, keepdim=True))     # keeps the total prob 1

            idx_next_pos = torch.multinomial(probs, num_samples=1)
            starting_text = torch.cat((starting_text, sorted_idx[0][idx_next_pos[0, 0]].unsqueeze(dim=0).unsqueeze(dim=0)), dim=1)

            if end_token != -1 and sorted_idx[0, idx_next_pos[0, 0]] == end_token:
                print("End token hit, stopping...")
                break
        
        return self.format_output(starting_text)

    def beam_search(self, starting_text, max_new_tokens, k=2, end_token=None):
        beams = [(starting_text, 0)]  # Initialize beams as a tuple of (sequence, score)

        for _ in tqdm(range(max_new_tokens), desc="Generating tokens"):
            new_beams = []
            for beam in beams:
                beam_seq, beam_score = beam
                logits, loss = self.model(beam_seq[:, -self.model_block_size:])
                logits = logits[:, -1, :]

                top_k_logits, top_k_idx = torch.topk(logits, k)

                for i in range(k):
                    token_logits = top_k_logits[0][i]
                    token_idx = top_k_idx[0][i]

                    new_seq = torch.cat((beam_seq, token_idx.unsqueeze(dim=0).unsqueeze(dim=0)), dim=1)
                    new_score = beam_score + token_logits

                    if end_token != -1 and token_idx == end_token:
                        return self.format_output(new_seq)

                    new_beams.append((new_seq, new_score))

            # Sort all new beams and select the top k
            beams = sorted(new_beams, key=lambda x: x[1], reverse=True)[:k]

        # Return the best beam
        best_seq, best_score = beams[0]
        return self.format_output(best_seq)

    def generate(self, starting_text, max_new_tokens, end_token=None, mode=0):
        self.model.eval()

        with torch.no_grad():
            if mode == 0:
                return self.top_p(starting_text, max_new_tokens, end_token)
            elif mode == 1:
                return self.random_sample(starting_text, max_new_tokens, end_token)
            elif mode == 2:
                return self.beam_search(starting_text, max_new_tokens, end_token)
            else:
                return self.top_p(starting_text, max_new_tokens, end_token)
