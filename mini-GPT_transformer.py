# 要读懂这些代码的话，最顺畅的路线是从GPTLanguageModel类的forward方法开始读起，理解它是如何调用其他模块的。然后再去看每个模块的实现细节，最后再回到GPTLanguageModel类的generate方法，看看它是如何利用前向传播来生成文本的。

import torch
import torch.nn as nn
from torch.nn import functional as F

# hyperparameters
batch_size = 64 # how many independent sequences will we process in parallel?
block_size = 256 # what is the maximum context length for predictions?
max_iters = 5000 # 代表训练多少次迭代。每次迭代都会从训练数据中抽取一个batch进行训练
eval_interval = 500 # 每隔多少次迭代评估一次模型在训练集和验证集上的损失。
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu' # 选择设备，如果有GPU可用则使用GPU，否则使用CPU
eval_iters = 200
n_embd = 384
n_head = 6
n_layer = 6
dropout = 0.2
# ------------

torch.manual_seed(1337)

# wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
with open('input.txt', 'r', encoding='utf-8') as f: 
    text = f.read()

# here are all the unique characters that occur in this text
chars = sorted(list(set(text)))
vocab_size = len(chars)
# create a mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

# Train and test splits
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data)) # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]

# data loading
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device) # 将数据移动到设备上（GPU或CPU），以便后续的模型训练和评估能够在正确的设备上进行。
    return x, y

@torch.no_grad()
def estimate_loss(): # 评估模型在训练集和验证集上的损失。该函数在评估过程中不计算梯度，以节省内存和计算资源。函数会返回一个字典，包含训练集和验证集的平均损失。
    out = {}
    model.eval() # 将模型设置为评估模式
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters) 
        for k in range(eval_iters): # 评估eval_iters次，每次都从数据集中抽取一个batch进行评估，并计算损失。最后将所有评估的损失取平均，得到训练集和验证集的平均损失。
            X, Y = get_batch(split)  
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, head_size): 
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size))) # register_buffer是一个特殊的方法，用于在模型中注册一个持久化的缓冲区（buffer）。这个缓冲区不会被视为模型的参数，因此在训练过程中不会被更新，但它会随着模型一起保存和加载。在这里，我们使用register_buffer来注册一个下三角矩阵tril，这个矩阵用于在计算自注意力时进行掩码操作，确保每个时间步只能关注之前的时间步，从而实现因果关系。 

        self.dropout = nn.Dropout(dropout) 

    def forward(self, x):
        # input of size (batch, time-step, channels)
        # output of size (batch, time-step, head-size) head_size是每个注意力头的维度，通常是嵌入维度除以注意力头的数量。这个输出表示每个时间步在该注意力头上的表示。
        B,T,C = x.shape # B是batch size，T是时间步数，C是嵌入维度
        k = self.key(x)   # (B,T,head-size)。(B,T,C) @ (C, head-size) -> (B,T,head-size)
        q = self.query(x) # (B,T,head-size)
        # compute attention scores ("affinities")
        wei = q @ k.transpose(-2,-1) * k.shape[-1]**-0.5 # (B, T, head-size) @ (B, head-size, T) -> (B, T, T) 
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T) 
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        wei = self.dropout(wei) 
        # perform the weighted aggregation of the values
        v = self.value(x) # (B,T,head-size)
        out = wei @ v # (B, T, T) @ (B, T, head-size) -> (B, T, head-size)
        return out

class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)]) # 创建一个ModuleList，包含num_heads个Head实例，每个Head实例的head_size为head_size。ModuleList是一个特殊的容器，用于存储一系列的子模块（在这里是多个注意力头）。通过使用ModuleList，我们可以方便地管理和调用这些子模块。
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1) # 对每个注意力头进行前向传播，得到每个头的输出，然后在最后一个维度上将它们连接起来，形成一个新的张量out。这个张量的形状是(B, T, head_size * num_heads)，因为我们把num_heads个头的输出连接在一起了。
        out = self.dropout(self.proj(out))
        return out

class FeedFoward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd), # 这个4是Transformer中常用的一个超参数，表示在前馈网络中隐藏层的维度是输入维度的4倍。这个设计是为了增加模型的表达能力，使其能够捕捉更复杂的模式和关系。
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout), # Dropout会在训练过程中随机地将一部分神经元的输出设置为零，从而减少模型对特定神经元的依赖，提高模型的泛化能力。
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module): 
    """ Transformer block: communication followed by computation """

    def __init__(self, n_embd, n_head):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = n_embd // n_head  # 384/6=64
        self.sa = MultiHeadAttention(n_head, head_size) # self-attention
        self.ffwd = FeedFoward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd) 
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x)) # 加法代表一个残差连接
        x = x + self.ffwd(self.ln2(x)) # feedforward之前进行layernorm，与最初的transformer的设计不同的地方，最初的transformer是在每个子层（self-attention和feedforward）之后进行layernorm的，而这里是在每个子层的输入之前进行layernorm的。这种设计被称为Pre-LN（Pre-LayerNorm），相对于Post-LN（Post-LayerNorm）来说，Pre-LN在训练深层Transformer时更稳定，能够更好地传播梯度，从而提高模型的性能和训练效率。
        return x

class GPTLanguageModel(nn.Module):

    def __init__(self):
        super().__init__() 
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd) # 词嵌入表
        self.position_embedding_table = nn.Embedding(block_size, n_embd) # 位置嵌入表，block_size是最大上下文长度，n_embd是嵌入维度
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)]) # *的作用是将列表中的元素作为位置参数传递给函数。在这里，我们创建了一个包含n_layer（6）个Block实例的列表，然后使用*将这个列表中的Block实例作为参数传递给nn.Sequential，构建了一个由多个Transformer块组成的序列模型。
        self.ln_f = nn.LayerNorm(n_embd) # final layer norm
        self.lm_head = nn.Linear(n_embd, vocab_size) # 线性层，将嵌入维度映射到词汇表大小，以便输出每个位置的下一个字符的概率分布

        # better init, not covered in the original GPT video, but important, will cover in followup video
        self.apply(self._init_weights)

    def _init_weights(self, module): # 这个函数是用来初始化模型权重的。对于线性层（nn.Linear），我们使用正态分布进行权重初始化，均值为0，标准差为0.02。如果线性层有偏置项，我们将其初始化为零。对于嵌入层（nn.Embedding），我们也使用正态分布进行权重初始化，均值为0，标准差为0.02。这种初始化方法有助于模型更快地收敛，并且在训练过程中表现更好。
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx) # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T,C),每个时间步位置赋予一个位置嵌入
        x = tok_emb + pos_emb # (B,T,C)，将token嵌入和位置嵌入相加，得到每个位置的输入表示
        x = self.blocks(x) # (B,T,C)
        x = self.ln_f(x) # (B,T,C)
        logits = self.lm_head(x) # # logits形状为(B, T, vocab_size)，其中的每一项（[b,t]）代表了从该样本的该时间步往后预测下一个字符的概率分布

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets) # 用来对比的targets是idx的下一个时间步的字符索引，启示本质上就是这一个样本往右移动一个时间步的版本，见mini-GPT.ipynb中get_batch的例子

        return logits, loss

    def generate(self, idx, max_new_tokens): 
        # 用于推理阶段，即根据当前上下文生成新的文本。主要作用是根据给定的输入idx（当前上下文）生成新的文本。它通过循环迭代max_new_tokens次，每次生成一个新的token，并将其添加到输入idx中，形成新的上下文，继续生成下一个token。这个过程会一直持续，直到生成了指定数量的新token为止。最终返回的是包含原始输入和新生成token的完整序列。
        
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -block_size:]  # 由于模型的最大上下文长度是block_size，所以在生成新token时，我们只考虑当前上下文的最后block_size个token作为输入。这是为了确保输入的长度不会超过模型的限制，同时也能让模型关注到最近的上下文信息，从而生成更相关的内容。
            # get the predictions
            logits, loss = self(idx_cond) # logits形状为(B, T, vocab_size)，其中的每一项（[b,t]）代表了从该样本的该时间步往后预测下一个字符的概率分布。self(idx_cond)调用了模型的前向传播方法
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, vocab_size)，只取最后一个时间步的logits，因为我们只需要根据当前上下文预测下一个字符
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, vocab_size)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx



# 开始训练

model = GPTLanguageModel()
m = model.to(device)
# print the number of parameters in the model
print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters') # .numel()方法返回一个张量中元素的总数，也就是参数的数量。计算模型的参数数量，并将其除以1e6以得到以百万为单位的参数数量。这个信息对于了解模型的规模和复杂度非常有用，尤其是在比较不同模型或评估模型的资源需求时。参数量大约10M


# create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):

    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # sample a batch of data
    xb, yb = get_batch('train')

    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step() # 通过调用optimizer.step()，我们更新模型的参数，使其朝着最小化损失的方向前进

# generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))
#open('more.txt', 'w').write(decode(m.generate(context, max_new_tokens=10000)[0].tolist()))