# NN For Miniworld and Miliastra Wonderland

A single-file neural network library for Lua 5.4. Zero dependencies. Implements
autograd, a full module system, optimizers, and transformer layers — everything
needed to build and train MLPs, CNNs, RNNs, and transformers in pure Lua.

Run the self-test: `lua nn.lua`  
Run the demo: `lua demo.lua`

---

## Usage

### Append style (no prefix needed)

Copy `nn.lua` into your project and append your code directly after the exports
block. Every class (`Tensor`, `Linear`, `Adam`, …) is already in scope as a local.

```lua
-- at the bottom of nn.lua:
manualSeed(42)
local model = Sequential.new(
  Linear.new(4, 16),
  ReLU.new(),
  Linear.new(16, 2)
)
```

### Require style

```lua
local nn = require("nn")
local Tensor = nn.Tensor

nn.manualSeed(42)
local model = nn.Sequential.new({
  nn.Linear.new(4, 16),
  nn.ReLU.new(),
  nn.Linear.new(16, 2),
})
```



---

## Quick example — MLP for sin(x)

```lua
local nn     = require("nn")
local Tensor = nn.Tensor

nn.manualSeed(42)

-- 64 training points over [-π, π]
local N, pi = 64, math.pi
local X = Tensor.new({N, 1})
local Y = Tensor.new({N, 1})
for i = 1, N do
  local x = -pi + (i - 1) * (2 * pi / (N - 1))
  X.data[i] = x
  Y.data[i] = math.sin(x)
end

local model = nn.Sequential.new({
  nn.Linear.new(1, 32), nn.Tanh.new(),
  nn.Linear.new(32, 32), nn.Tanh.new(),
  nn.Linear.new(32, 1),  nn.Tanh.new(),
})

local loss_fn = nn.MSELoss.new()
local opt     = nn.Adam.new(model:parameters(), 3e-3)

for epoch = 1, 800 do
  model:zeroGrad()
  local loss = loss_fn(model:forward(X), Y)
  loss:backward()
  opt:step()
end
```

See `demo.lua` for the full version with evaluation output.

---

## What's included

| Category | Classes |
|---|---|
| **Core** | `Tensor`, `Module`, `Sequential` |
| **Layers** | `Linear`, `Embedding` |
| **Activations** | `ReLU`, `Sigmoid`, `Tanh`, `GELU`, `LeakyReLU` |
| **Normalization** | `LayerNorm`, `BatchNorm1d`, `Dropout` |
| **Loss** | `MSELoss`, `L1Loss`, `BCELoss`, `CrossEntropyLoss` |
| **Optimizers** | `SGD`, `Adam`, `AdamW`, `RMSprop`, `Adagrad` |
| **Schedulers** | `StepLR`, `CosineAnnealingLR` |
| **Recurrent** | `RNN`, `LSTM`, `GRU` |
| **Convolutional** | `Conv1d`, `Conv2d`, `MaxPool1d`, `MaxPool2d`, `AvgPool2d` |
| **Transformer** | `MultiHeadAttention`, `TransformerEncoderLayer`, `TransformerEncoder`, `TransformerDecoderLayer`, `TransformerDecoder` |
| **Utilities** | `manualSeed`, `gradcheck`, `Tensor.randn/rand/ones/zeros`, `Tensor.fromTable` |

---

## Tensor

The core data type. All arithmetic builds a computation graph when
`requires_grad` is true; call `:backward()` on a scalar output to populate
`.grad` on every leaf.

```lua
local t = Tensor.new({3, 4})          -- zero-initialised [3, 4]
local t = Tensor.new({3, 4}, 1.0)     -- filled with 1.0
local t = Tensor.new({3}, {1, 2, 3})  -- from a flat table
local t = Tensor.randn({3, 4})        -- N(0,1)
local t = Tensor.fromTable({{1,2},{3,4}})  -- from a nested table

t.requires_grad = true
local out = t:matmul(other):sum()
out:backward()
-- t.grad now holds ∂out/∂t
```

Shape ops: `reshape`, `transpose`, `permute`, `unsqueeze`, `squeeze`, `view`.  
Reductions: `sum`, `mean`, `max` — all accept an optional `dim` argument.  
Arithmetic: `add`, `sub`, `mul`, `div`, `pow`, `neg`, `exp`, `log`, `sqrt`,
`abs`, `matmul` — all broadcast and propagate gradients.

---

## Modules

Every layer shares the same interface:

```lua
layer:forward(x)    -- returns output tensor; callable as layer(x) too
layer:parameters()  -- flat table of all parameter tensors
layer:zeroGrad()    -- zero .grad on every parameter
layer:train()       -- training mode (affects Dropout, BatchNorm1d)
layer:eval()        -- eval mode
```

### Linear

```lua
nn.Linear.new(inFeatures, outFeatures, bias?)  -- bias defaults to true
```

Accepts `[B, inF]` or `[B, T, inF]` input; returns the same rank with the last
dimension projected to `outFeatures`.

### Embedding

```lua
nn.Embedding.new(numEmbeddings, embDim)
-- input: flat Lua table of 1-based integer indices (or a 1-D Tensor)
-- output: [len, embDim]
```

### Normalization and regularization

```lua
nn.LayerNorm.new({dModel})           -- normalizes over last N dims
nn.BatchNorm1d.new(numFeatures)      -- [B,C] or [B,C,L] input
nn.Dropout.new(p)                    -- drop probability p, default 0.5
```

---

## Loss functions

Losses are instantiated once, then called like functions. The target may be a
flat Lua table of numbers or a `Tensor` with the same shape as the prediction.

```lua
local ce   = nn.CrossEntropyLoss.new()
local loss = ce(logits, {2, 1, 3})  -- 1-based class labels
loss:backward()
```

| Loss | Notes |
|---|---|
| `MSELoss` | `mean((pred - target)²)` |
| `L1Loss` | mean absolute error |
| `BCELoss` | binary cross-entropy; pred must be in (0, 1) |
| `CrossEntropyLoss` | log-softmax + NLL; logits are `[B, C]`, labels are 1-based |

All four accept an optional `reduction` string (`"mean"` or `"sum"`) passed to
`.new()`.

---

## Optimizers

```lua
nn.SGD.new(params, lr, momentum?, weightDecay?)
nn.Adam.new(params, lr, betas?, eps?, weightDecay?)
  -- betas defaults to {0.9, 0.999}, eps to 1e-8
nn.AdamW.new(params, lr, betas?, eps?, weightDecay?)
nn.RMSprop.new(params, lr, alpha?, eps?, momentum?)
nn.Adagrad.new(params, lr, eps?)

opt:step()      -- update parameters from their .grad
opt:zeroGrad()  -- zero all .grad fields
```

`params` is the flat table returned by `model:parameters()`.

## LR schedulers

```lua
local sched = nn.StepLR.new(opt, {stepSize = 10, gamma = 0.5})
local sched = nn.CosineAnnealingLR.new(opt, {T_max = 100, eta_min = 1e-6})

sched:step()  -- call once per epoch, after opt:step()
```

---

## Recurrent layers

RNN, LSTM, and GRU all use batched input with the batch dimension second:
`[T, B, inputSize]`.

```lua
nn.RNN.new(inputSize, hiddenSize, numLayers?)
output, hN = rnn:forward(x)
-- x:      [T, B, inputSize]
-- output: [T, B, hiddenSize]
-- hN:     table of [B, hiddenSize] per layer

nn.LSTM.new(inputSize, hiddenSize, numLayers?)
output, h = lstm:forward(x)  -- h is the final hidden state table

nn.GRU.new(inputSize, hiddenSize, numLayers?)
output, hN = gru:forward(x)
```

---

## Convolutional layers

```lua
nn.Conv2d.new(inC, outC, kernelSize, stride?, padding?)
-- kernelSize: integer or {kH, kW}; input [B, inC, H, W]

nn.Conv1d.new(inC, outC, kernelSize, stride?, padding?)
-- input [B, inC, L]

nn.MaxPool2d.new(kernelSize, stride?)   -- input [B, C, H, W]
nn.AvgPool2d.new(kernelSize, stride?)   -- input [B, C, H, W]
nn.MaxPool1d.new(kernelSize, stride?)   -- input [B, C, L]
-- stride defaults to kernelSize (non-overlapping)
```

---

## Transformer layers

All shapes are `[B, T, D]`. Indices are 1-based throughout.

```lua
nn.MultiHeadAttention.new(embedDim, numHeads, dropout?)
out = mha:forward(q, k, v, attn_mask?)
-- q: [B, Tq, D], k/v: [B, Tk, D]
-- attn_mask: additive [1,1,Tq,Tk] or [B,H,Tq,Tk]; -math.huge masks a position
-- out: [B, Tq, D]

nn.TransformerEncoderLayer.new(dModel, nHead, dimFeedforward?, dropout?)
-- dimFeedforward defaults to 4*dModel, dropout to 0.1
-- out = layer:forward(x)  -- x and out: [B, T, D]

nn.TransformerEncoder.new(encoderLayer, numLayers)
-- stacks numLayers independent copies of encoderLayer

nn.TransformerDecoderLayer.new(dModel, nHead, dimFeedforward?, dropout?)
-- out = layer:forward(tgt, memory)
-- applies causal mask on self-attention automatically

nn.TransformerDecoder.new(decoderLayer, numLayers)
-- out = decoder:forward(tgt, memory)
```

### Sequence parity example

```lua
-- append after nn.lua, or use nn. prefix in require style

manualSeed(42)
local VOCAB, SEQ, D, NHEAD = 8, 4, 8, 2

local emb     = Embedding.new(VOCAB, D)
local encoder = TransformerEncoder.new(
                  TransformerEncoderLayer.new(D, NHEAD, 32, 0), 1)
local head    = Linear.new(SEQ * D, 2)

local params = {}
for _, mod in ipairs({emb, encoder, head}) do
  for _, p in ipairs(mod:parameters()) do params[#params+1] = p end
end

local opt = Adam.new(params, 2e-4)
local ce  = CrossEntropyLoss.new()

opt:zeroGrad()
local x = emb:forward({1, 3, 5, 2}):unsqueeze(1)  -- [1, 4, 8]
x = encoder:forward(x)                              -- [1, 4, 8]
x = x:reshape({1, SEQ * D})                        -- [1, 32]
local loss = ce(head:forward(x), {1})
loss:backward()
opt:step()
```

---

## Utilities

```lua
nn.manualSeed(seed)     -- seed math.random for reproducibility

nn.gradcheck(fn, inputs, opts?)
-- numerical gradient check; fn() must return a scalar tensor
-- inputs: table of tensors with requires_grad = true
-- opts: { eps = 1e-4, tol = 1e-2 }
-- returns max_abs_error, max_rel_error
```

`Tensor.fromTable(tbl)` builds a tensor from a nested Lua table, inferring the
shape from the nesting depth and lengths.

---

## Notes

**Indexing** — everything is 1-based, matching Lua convention.

**No GPU** — all computation runs on the CPU. The library is only designed for
Miniworld and Miliastra Wonderland; large models will be slow.

**No in-place autograd** — `fill`, `zero_`, and direct writes to `t.data` are
not tracked. Use them only on freshly allocated tensors or after `:detach()`.

**No parameter sharing** — `TransformerEncoder` and `TransformerDecoder` create
independent weight copies for each layer beyond the first. For shared weights,
pass `numLayers = 1` and call `forward` manually in a loop.

---

## Files

| File | Purpose |
|---|---|
| `nn.lua` | The library; also a self-test when run directly (`lua nn.lua`) |
| `demo.lua` | Trains a small MLP to approximate sin(x) |
| `transformer.lua` | Transformer usage example |
| `test.lua` / `test_nn.lua` | Integration test suites |
| `doc.md` | Full API reference |

Generated by Claude Opus 5


