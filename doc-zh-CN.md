# nn.lua — 纯 Lua 神经网络库

零依赖，适用于 Lua 5.4。实现了自动微分、模块、优化器和 Transformer 层。

## 用法

由于千星奇域或迷你世界平台限制，请将代码直接追加到 `nn.lua`末尾。所有名称（`Tensor`、`Linear`、`Adam`、……）都在局部作用域内——不需要 `nn.` 前缀。

```lua
-- at the bottom of nn.lua, after the exports section:

manualSeed(42)
local model = Sequential.new(
  Linear.new(4, 16),
  ReLU.new(),
  Linear.new(16, 2)
)
```


---

## 张量

核心数据类型。当 `requires_grad` 为 true 时，所有算术运算都会构建计算图。

### 构造函数

```lua
Tensor.new(shape, data?)     -- zero-initialised, or from a flat table
Tensor.randn(shape)          -- samples from N(0,1)
Tensor.rand(shape)           -- uniform [0, 1)
Tensor.ones(shape)           -- filled with 1
Tensor.zeros(shape)          -- filled with 0
```

`shape` 是一个由整数组成的 Lua 表，例如 `{3, 4}` 表示一个 3×4 矩阵。`data` 是一个可选的扁平表；超出张量大小的元素会被忽略，缺失的元素会变为 0。

### 属性

| 属性 | 类型 | 描述 |
|---|---|---|
| `t.shape` | table | 维度大小，例如 `{2, 3}` |
| `t.data` | table | 按行优先排列的一维存储，索引从 1 开始 |
| `t.requires_grad` | bool | 设为 true 以追踪梯度 |
| `t.grad` | Tensor 或 nil | 执行 `backward()` 后累积的梯度 |

### 方法

```lua
t:numel()            -- total number of elements
t:ndim()             -- number of dimensions
t:clone()            -- deep copy, same requires_grad
t:detach()           -- copy with requires_grad=false and no grad_fn
t:fill(v)            -- fill every element with v, returns self
t:zero_()            -- fill with 0, returns self
t:item()             -- return the single scalar value (errors if numel > 1)
t:zero_grad()        -- set t.grad = nil
```

### 算术运算

所有运算符都支持广播，并且当任一操作数具有 `requires_grad = true` 时会传播梯度。

```lua
a:add(b)             -- a + b  (also works with a scalar number)
a:sub(b)             -- a - b
a:mul(b)             -- element-wise multiply, or scalar multiply
a:pow(n)             -- element-wise power
a:neg()              -- negate
a:matmul(b)          -- matrix multiply; b must be 2-D
a:transpose(d1, d2)  -- swap two dimensions
a:reshape(shape)     -- new shape (same numel)
a:unsqueeze(dim)     -- insert a size-1 dimension at dim
a:squeeze(dim?)      -- remove size-1 dimension at dim, or all of them
```

张量级激活函数（模块版本的别名，但可直接对张量调用）：

```lua
t:relu()
t:sigmoid()
t:tanh()
```

### 自动微分

```lua
t.requires_grad = true   -- enable gradient tracking before use

out:backward()           -- backprop from a scalar output; seeds with ones
out:backward(g)          -- backprop from a non-scalar, g has the same shape as out
```

执行 `backward()` 后，`t.grad` 会保存每个具有 `requires_grad = true` 的叶张量累积的梯度。在下一次前向传播之前，调用 `t:zero_grad()` 或优化器的 `zeroGrad()`。

### 实用函数

这些是模块级函数，不是方法：

```lua
cat(tensors, dim)        -- concatenate a table of tensors along dim (1-based)
stack(tensors, dim)      -- stack a table of tensors along a new dimension dim
manualSeed(seed)         -- seed math.random for reproducibility
gradcheck(fn, inputs, opts?)
  -- numerical gradient check; fn() must return a scalar tensor
  -- inputs: table of tensors with requires_grad=true
  -- opts: { eps=1e-4, tol=1e-2 }
  -- returns max_abs_error, max_rel_error
```

---

## Module 基类

每一层都继承自 `Module`。它们的接口都相同：

```lua
layer:forward(x)       -- run the layer, returns output tensor
layer:parameters()     -- returns a flat table of all leaf parameter tensors
layer:train()          -- set training mode (affects Dropout, BatchNorm)
layer:eval()           -- set eval mode
layer:zeroGrad()       -- zero .grad on every parameter
```

将实例作为函数调用，等同于调用 `:forward`：

```lua
local y = linear(x)    -- equivalent to linear:forward(x)
```

---

## 层

### 线性层

```lua
Linear.new(inFeatures, outFeatures, bias?)
```

`bias` 默认为 `true`。接受任意形状 `[*, inFeatures]` 的输入，并返回 `[*, outFeatures]`。可直接处理二维 `[B, inF]` 和三维 `[B, T, inF]` 输入，无需手动重塑。

参数：`weight``[outF, inF]`、`bias``[outF]`。

### 嵌入层

```lua
Embedding.new(numEmbeddings, embDim)
```

从权重矩阵中查找行。输入为一维 Lua 表，其中包含从 1 开始的整数索引（或一维 Tensor）。返回 `[len(indices), embDim]`。梯度按索引累积到 `weight`中。

参数：`weight``[numEmbeddings, embDim]`。

### 顺序容器

```lua
Sequential.new(layer1, layer2, ...)
Sequential.new({layer1, layer2, ...})   -- table form
seq:add(layer)                          -- append a layer, returns self
```

按顺序运行各层；每一层的输出都是下一层的输入。

---

## 激活函数

所有激活函数都是无状态模块（没有参数）。也可以作为张量方法使用。

```lua
ReLU.new()
Sigmoid.new()
Tanh.new()
GELU.new()           -- approximate GELU (tanh form)
LeakyReLU.new(negativeSlope?)   -- default slope 0.01
```

---

## 归一化

### 层归一化

```lua
LayerNorm.new(normalizedShape)   -- e.g. LayerNorm.new({128})
```

对最后 `#normalizedShape` 个维度进行归一化。参数：`weight`和`bias`，二者形状均与`normalizedShape`相同，初始值分别为全 1 和全 0。

### 批归一化1d

```lua
BatchNorm1d.new(numFeatures, eps?, momentum?)
```

接受 `[B, C]` 或 `[B, C, L]` 输入。训练模式下使用批次统计量，并更新 `runningMean` / `runningVar`；评估模式下使用运行统计量。`eps` 默认值为 `1e-5`，`momentum` 默认值为 `0.1`。

参数：`weight``[C]`、`bias``[C]`。

### 随机失活

```lua
Dropout.new(p?)   -- p is drop probability, default 0.5
```

训练模式下会随机将元素置零，并将保留的元素缩放 `1/(1-p)`。评估模式下不执行任何操作。

---

## 损失函数

损失函数只需构造一次，之后即可像函数一样调用。目标值始终是普通的 Lua 表。

```lua
local loss_fn = MSELoss.new()
local loss     = loss_fn(prediction, target_table)
loss:backward()
```

### 均方误差损失

```lua
MSELoss.new()
-- loss_fn(pred, target)
-- pred:   [B, *]  tensor
-- target: flat table of numbers, same numel as pred
```

均方误差：`mean((pred - target)^2)`。

### L1损失

```lua
L1Loss.new()
-- loss_fn(pred, target)
```

平均绝对误差。

### 二元交叉熵损失

```lua
BCELoss.new()
-- loss_fn(pred, target)
-- pred values must be in (0, 1)
-- target values must be 0 or 1
```

用于 sigmoid 输出的二元交叉熵。

### 交叉熵损失

```lua
CrossEntropyLoss.new()
-- loss_fn(logits, labels)
-- logits: [B, numClasses]  (raw, unnormalized scores)
-- labels: table of 1-based class indices, length B
```

先应用 log-softmax，再计算 NLL 损失。数值稳定。

---

## 优化器

所有优化器都使用相同的接口：

```lua
opt:step()       -- update parameters using their current .grad
opt:zeroGrad()   -- zero .grad on all tracked parameters
```

### 随机梯度下降

```lua
SGD.new(params, lr, momentum?, weightDecay?)
```

标准 SGD，可选动量和 L2 权重衰减。

### Adam

```lua
Adam.new(params, lr, betas?, eps?, weightDecay?)
-- betas default {0.9, 0.999}, eps default 1e-8
```

### AdamW

```lua
AdamW.new(params, lr, betas?, eps?, weightDecay?)
```

采用解耦权重衰减的 Adam（权重衰减直接应用于参数，而不是通过梯度应用）。

### RMSprop

```lua
RMSprop.new(params, lr, alpha?, eps?, momentum?)
-- alpha default 0.99, eps default 1e-8
```

### Adagrad

```lua
Adagrad.new(params, lr, eps?)
-- eps default 1e-10
```

---

## 学习率调度器

调度器封装一个优化器，并调整其 `lr` 字段。每个 epoch 调用一次 `sched:step()`（在 `opt:step()` 之后）。

### 阶梯学习率调度器

```lua
StepLR.new(optimizer, stepSize, gamma?)
-- gamma defaults to 0.1
-- multiplies lr by gamma every stepSize epochs
```

### 余弦退火学习率调度器

```lua
CosineAnnealingLR.new(optimizer, tMax, etaMin?)
-- etaMin defaults to 0
-- anneals lr from its initial value to etaMin over tMax steps
```

---

## 循环层

### 循环神经网络

```lua
RNN.new(inputSize, hiddenSize, numLayers?, dropout?)
output, hN = rnn:forward(x, h0?)
-- x:   [T, inputSize]
-- h0:  [numLayers, hiddenSize]  (optional, defaults to zeros)
-- output: [T, hiddenSize]
-- hN:     [numLayers, hiddenSize]
```

### 长短期记忆网络

```lua
LSTM.new(inputSize, hiddenSize, numLayers?, dropout?)
output, state = lstm:forward(x, state?)
-- x:     [T, inputSize]
-- state: {h0, c0}  each [numLayers, hiddenSize]  (optional)
-- output: [T, hiddenSize]
-- state:  {hN, cN}
```

### 门控循环单元

```lua
GRU.new(inputSize, hiddenSize, numLayers?, dropout?)
output, hN = gru:forward(x, h0?)
-- same shape conventions as RNN
```

---

## 卷积层

### 二维卷积

```lua
Conv2d.new(inChannels, outChannels, kernelSize, stride?, padding?)
-- kernelSize: integer (square) or {kH, kW}
-- stride, padding: integer or {H, W}, default 1 and 0
-- input:  [B, inC, H, W]
-- output: [B, outC, H_out, W_out]
```

### 一维卷积

```lua
Conv1d.new(inChannels, outChannels, kernelSize, stride?, padding?)
-- input:  [B, inC, L]
-- output: [B, outC, L_out]
```

### 池化

```lua
MaxPool2d.new(kernelSize, stride?)    -- input [B,C,H,W]
AvgPool2d.new(kernelSize, stride?)    -- input [B,C,H,W]
MaxPool1d.new(kernelSize, stride?)    -- input [B,C,L]
```

`stride` defaults to `kernelSize` (non-overlapping).

---

## Transformer 层

所有 Transformer 形状都使用从 1 开始的索引。批次维度 B 始终位于最前面。

### 多头注意力

```lua
MultiHeadAttention.new(embedDim, numHeads, dropout?)
out = mha:forward(q, k, v, attn_mask?)
-- q:         [B, Tq, D]
-- k, v:      [B, Tk, D]
-- attn_mask: optional additive mask [1,1,Tq,Tk] or [B,H,Tq,Tk]
--            use -math.huge for positions to ignore (e.g. causal mask)
-- out:       [B, Tq, D]
```

原生支持交叉注意力（`Tq ≠ Tk`）。参数：`Wq`、`Wk`、`Wv`、`Wo`，每个`[D, D]`。

### TransformerEncoderLayer

```lua
TransformerEncoderLayer.new(dModel, nHead, dimFeedforward?, dropout?)
-- dimFeedforward defaults to 4*dModel, dropout to 0.1
out = layer:forward(x)
-- x:   [B, T, D]
-- out: [B, T, D]
```

自注意力 → 残差连接 + LayerNorm → 前馈网络（Linear→ReLU→Linear）→ 残差连接 + LayerNorm。

### TransformerEncoder

```lua
TransformerEncoder.new(encoderLayer, numLayers)
out = encoder:forward(x)
-- x:   [B, T, D]
-- out: [B, T, D]
```

堆叠`numLayers`个相互独立的`encoderLayer`副本（第一层是传入的层；后续层使用相同维度重新初始化）。

### TransformerDecoderLayer

```lua
TransformerDecoderLayer.new(dModel, nHead, dimFeedforward?, dropout?)
out = layer:forward(tgt, memory)
-- tgt:    [B, Tq, D]  decoder input sequence
-- memory: [B, Tk, D]  encoder output
-- out:    [B, Tq, D]
```

掩码自注意力（自动应用因果掩码）→ 残差连接 + LayerNorm → 对`memory`进行交叉注意力 → 残差连接 + LayerNorm → 前馈网络 → 残差连接 + LayerNorm。

### TransformerDecoder

```lua
TransformerDecoder.new(decoderLayer, numLayers)
out = decoder:forward(tgt, memory)
-- tgt:    [B, Tq, D]
-- memory: [B, Tk, D]
-- out:    [B, Tq, D]
```

---

## 完整示例 — 序列奇偶分类器

```lua
-- append this after nn.lua

manualSeed(42)

local VOCAB, SEQ, D, NHEAD, D_FF = 8, 4, 8, 2, 32

local emb     = Embedding.new(VOCAB, D)
local encoder = TransformerEncoder.new(
                  TransformerEncoderLayer.new(D, NHEAD, D_FF, 0), 1)
local head    = Linear.new(SEQ * D, 2)

local params = {}
for _, p in ipairs(emb:parameters())     do params[#params+1] = p end
for _, p in ipairs(encoder:parameters()) do params[#params+1] = p end
for _, p in ipairs(head:parameters())    do params[#params+1] = p end

local opt = Adam.new(params, 2e-4)
local ce  = CrossEntropyLoss.new()

local function forward(tokens)
  local x = emb:forward(tokens):unsqueeze(1)   -- [1, 4, 8]
  x = encoder:forward(x)                        -- [1, 4, 8]
  x = x:reshape({1, SEQ * D})                  -- [1, 32]
  return head:forward(x)                        -- [1, 2]
end

-- single training step
opt:zeroGrad()
local logits = forward({1, 3, 5, 2})
local loss   = ce(logits, {1})    -- label 1 = odd
loss:backward()
opt:step()
```

---

## 注意事项

**索引** — 所有索引均从 1 开始，与 Lua 约定一致。

**无 GPU** — 所有计算都在纯 Lua 中由 CPU 执行。对于大型模型和数据集，训练会比较缓慢；该库旨在用于特定平台。

**不共享参数** — `TransformerEncoder`和`TransformerDecoder`会为第一层之后的每一层创建独立的权重副本。如果需要共享权重，请传入同一个层对象并设置`numLayers = 1`，然后手动多次调用`encoder:forward`。

**原地操作** — `fill`、`zero_`以及直接写入`t.data`都不会被 autograd 跟踪。只能对新建张量或分离后的张量使用这些操作。

代码与英语文档由Claude Opus 5生成，中文文档由DeepSeek翻译

bug报告请提交至Issues