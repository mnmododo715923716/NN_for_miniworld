# nn.lua — Pure-Lua Neural Network Library

Zero dependencies, Lua 5.4. Implements autograd, modules, optimizers, and transformer layers.

## Usage

Because there is no separate package to install, append your code directly to the end of `nn.lua`. All names (`Tensor`, `Linear`, `Adam`, …) are in scope as locals — no `nn.` prefix required.

```lua
-- at the bottom of nn.lua, after the exports section:

manualSeed(42)
local model = Sequential.new(
  Linear.new(4, 16),
  ReLU.new(),
  Linear.new(16, 2)
)
```

If you prefer to use `require`, the module also returns `nn` with every class attached (`nn.Linear`, `nn.Adam`, etc.), but the rest of this document assumes the append style.

---

## Tensor

The core data type. All arithmetic operations build a computation graph when `requires_grad` is true.

### Constructors

```lua
Tensor.new(shape, data?)     -- zero-initialised, or from a flat table
Tensor.randn(shape)          -- samples from N(0,1)
Tensor.rand(shape)           -- uniform [0, 1)
Tensor.ones(shape)           -- filled with 1
Tensor.zeros(shape)          -- filled with 0
```

`shape` is a Lua table of integers, e.g. `{3, 4}` for a 3×4 matrix. `data` is an optional flat table; elements beyond the tensor's size are ignored, missing ones become 0.

### Properties

| Property | Type | Description |
|---|---|---|
| `t.shape` | table | dimension sizes, e.g. `{2, 3}` |
| `t.data` | table | flat row-major storage, 1-indexed |
| `t.requires_grad` | bool | set true to track gradients |
| `t.grad` | Tensor or nil | accumulated gradient after `backward()` |

### Methods

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

### Arithmetic

All operators support broadcasting and propagate gradients when either operand has `requires_grad = true`.

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

Tensor-level activations (aliases for the module versions, but callable on a tensor directly):

```lua
t:relu()
t:sigmoid()
t:tanh()
```

### Autograd

```lua
t.requires_grad = true   -- enable gradient tracking before use

out:backward()           -- backprop from a scalar output; seeds with ones
out:backward(g)          -- backprop from a non-scalar, g has the same shape as out
```

After `backward()`, `t.grad` holds the accumulated gradient for every leaf tensor that had `requires_grad = true`. Call `t:zero_grad()` or the optimizer's `zeroGrad()` before the next forward pass.

### Utility functions

These are module-level functions, not methods:

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

## Module base class

Every layer inherits from `Module`. The interface is the same for all of them:

```lua
layer:forward(x)       -- run the layer, returns output tensor
layer:parameters()     -- returns a flat table of all leaf parameter tensors
layer:train()          -- set training mode (affects Dropout, BatchNorm)
layer:eval()           -- set eval mode
layer:zeroGrad()       -- zero .grad on every parameter
```

Calling an instance as a function is the same as calling `:forward`:

```lua
local y = linear(x)    -- equivalent to linear:forward(x)
```

---

## Layers

### Linear

```lua
Linear.new(inFeatures, outFeatures, bias?)
```

`bias` defaults to `true`. Accepts input of any shape `[*, inFeatures]` and returns `[*, outFeatures]`. Works on 2-D `[B, inF]` and 3-D `[B, T, inF]` inputs without manual reshaping.

Parameters: `weight` `[outF, inF]`, `bias` `[outF]`.

### Embedding

```lua
Embedding.new(numEmbeddings, embDim)
```

Looks up rows from a weight matrix. Input is a flat Lua table of 1-based integer indices (or a 1-D Tensor). Returns `[len(indices), embDim]`. Gradients accumulate into `weight` by index.

Parameters: `weight` `[numEmbeddings, embDim]`.

### Sequential

```lua
Sequential.new(layer1, layer2, ...)
Sequential.new({layer1, layer2, ...})   -- table form
seq:add(layer)                          -- append a layer, returns self
```

Runs layers in order; the output of each is the input to the next.

---

## Activations

All activations are stateless modules (no parameters). They can also be used as tensor methods.

```lua
ReLU.new()
Sigmoid.new()
Tanh.new()
GELU.new()           -- approximate GELU (tanh form)
LeakyReLU.new(negativeSlope?)   -- default slope 0.01
```

---

## Normalization

### LayerNorm

```lua
LayerNorm.new(normalizedShape)   -- e.g. LayerNorm.new({128})
```

Normalizes over the last `#normalizedShape` dimensions. Parameters: `weight` and `bias`, both shaped like `normalizedShape`, initialized to ones and zeros respectively.

### BatchNorm1d

```lua
BatchNorm1d.new(numFeatures, eps?, momentum?)
```

Accepts `[B, C]` or `[B, C, L]` input. In training mode uses batch statistics and updates `runningMean` / `runningVar`; in eval mode uses the running stats. `eps` defaults to `1e-5`, `momentum` to `0.1`.

Parameters: `weight` `[C]`, `bias` `[C]`.

### Dropout

```lua
Dropout.new(p?)   -- p is drop probability, default 0.5
```

In training mode randomly zeros elements and scales surviving ones by `1/(1-p)`. In eval mode is a no-op.

---

## Loss functions

Losses are constructed once and then called like functions. The target is always a plain Lua table.

```lua
local loss_fn = MSELoss.new()
local loss     = loss_fn(prediction, target_table)
loss:backward()
```

### MSELoss

```lua
MSELoss.new()
-- loss_fn(pred, target)
-- pred:   [B, *]  tensor
-- target: flat table of numbers, same numel as pred
```

Mean squared error: `mean((pred - target)^2)`.

### L1Loss

```lua
L1Loss.new()
-- loss_fn(pred, target)
```

Mean absolute error.

### BCELoss

```lua
BCELoss.new()
-- loss_fn(pred, target)
-- pred values must be in (0, 1)
-- target values must be 0 or 1
```

Binary cross-entropy for sigmoid outputs.

### CrossEntropyLoss

```lua
CrossEntropyLoss.new()
-- loss_fn(logits, labels)
-- logits: [B, numClasses]  (raw, unnormalized scores)
-- labels: table of 1-based class indices, length B
```

Applies log-softmax then NLL loss. Numerically stable.

---

## Optimizers

All optimizers share the same interface:

```lua
opt:step()       -- update parameters using their current .grad
opt:zeroGrad()   -- zero .grad on all tracked parameters
```

### SGD

```lua
SGD.new(params, lr, momentum?, weightDecay?)
```

Vanilla SGD with optional momentum and L2 weight decay.

### Adam

```lua
Adam.new(params, lr, betas?, eps?, weightDecay?)
-- betas default {0.9, 0.999}, eps default 1e-8
```

### AdamW

```lua
AdamW.new(params, lr, betas?, eps?, weightDecay?)
```

Adam with decoupled weight decay (weight decay applied to parameters directly, not through the gradient).

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

## Learning rate schedulers

Schedulers wrap an optimizer and adjust its `lr` field. Call `sched:step()` once per epoch (after `opt:step()`).

### StepLR

```lua
StepLR.new(optimizer, stepSize, gamma?)
-- gamma defaults to 0.1
-- multiplies lr by gamma every stepSize epochs
```

### CosineAnnealingLR

```lua
CosineAnnealingLR.new(optimizer, tMax, etaMin?)
-- etaMin defaults to 0
-- anneals lr from its initial value to etaMin over tMax steps
```

---

## Recurrent layers

### RNN

```lua
RNN.new(inputSize, hiddenSize, numLayers?, dropout?)
output, hN = rnn:forward(x, h0?)
-- x:   [T, inputSize]
-- h0:  [numLayers, hiddenSize]  (optional, defaults to zeros)
-- output: [T, hiddenSize]
-- hN:     [numLayers, hiddenSize]
```

### LSTM

```lua
LSTM.new(inputSize, hiddenSize, numLayers?, dropout?)
output, state = lstm:forward(x, state?)
-- x:     [T, inputSize]
-- state: {h0, c0}  each [numLayers, hiddenSize]  (optional)
-- output: [T, hiddenSize]
-- state:  {hN, cN}
```

### GRU

```lua
GRU.new(inputSize, hiddenSize, numLayers?, dropout?)
output, hN = gru:forward(x, h0?)
-- same shape conventions as RNN
```

---

## Convolutional layers

### Conv2d

```lua
Conv2d.new(inChannels, outChannels, kernelSize, stride?, padding?)
-- kernelSize: integer (square) or {kH, kW}
-- stride, padding: integer or {H, W}, default 1 and 0
-- input:  [B, inC, H, W]
-- output: [B, outC, H_out, W_out]
```

### Conv1d

```lua
Conv1d.new(inChannels, outChannels, kernelSize, stride?, padding?)
-- input:  [B, inC, L]
-- output: [B, outC, L_out]
```

### Pooling

```lua
MaxPool2d.new(kernelSize, stride?)    -- input [B,C,H,W]
AvgPool2d.new(kernelSize, stride?)    -- input [B,C,H,W]
MaxPool1d.new(kernelSize, stride?)    -- input [B,C,L]
```

`stride` defaults to `kernelSize` (non-overlapping).

---

## Transformer layers

All transformer shapes use 1-based indexing. Batch dimension B is always first.

### MultiHeadAttention

```lua
MultiHeadAttention.new(embedDim, numHeads, dropout?)
out = mha:forward(q, k, v, attn_mask?)
-- q:         [B, Tq, D]
-- k, v:      [B, Tk, D]
-- attn_mask: optional additive mask [1,1,Tq,Tk] or [B,H,Tq,Tk]
--            use -math.huge for positions to ignore (e.g. causal mask)
-- out:       [B, Tq, D]
```

Supports cross-attention (`Tq ≠ Tk`) natively. Parameters: `Wq`, `Wk`, `Wv`, `Wo` each `[D, D]`.

### TransformerEncoderLayer

```lua
TransformerEncoderLayer.new(dModel, nHead, dimFeedforward?, dropout?)
-- dimFeedforward defaults to 4*dModel, dropout to 0.1
out = layer:forward(x)
-- x:   [B, T, D]
-- out: [B, T, D]
```

Self-attention → residual + LayerNorm → feedforward (Linear→ReLU→Linear) → residual + LayerNorm.

### TransformerEncoder

```lua
TransformerEncoder.new(encoderLayer, numLayers)
out = encoder:forward(x)
-- x:   [B, T, D]
-- out: [B, T, D]
```

Stacks `numLayers` independent copies of `encoderLayer` (first layer is the one passed in; subsequent layers are freshly initialized with the same dimensions).

### TransformerDecoderLayer

```lua
TransformerDecoderLayer.new(dModel, nHead, dimFeedforward?, dropout?)
out = layer:forward(tgt, memory)
-- tgt:    [B, Tq, D]  decoder input sequence
-- memory: [B, Tk, D]  encoder output
-- out:    [B, Tq, D]
```

Masked self-attention (causal mask applied automatically) → residual + LayerNorm → cross-attention over `memory` → residual + LayerNorm → feedforward → residual + LayerNorm.

### TransformerDecoder

```lua
TransformerDecoder.new(decoderLayer, numLayers)
out = decoder:forward(tgt, memory)
-- tgt:    [B, Tq, D]
-- memory: [B, Tk, D]
-- out:    [B, Tq, D]
```

---

## Full example — sequence parity classifier

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

## Notes

**Indexing** — everything is 1-based, matching Lua convention.

**No GPU** — all computation runs on the CPU in pure Lua. For large models and datasets, expect training to be slow; the library is designed for learning and experimentation.

**No parameter sharing** — `TransformerEncoder` and `TransformerDecoder` create independent weight copies for each layer beyond the first. If you need shared weights, pass the same layer object and set `numLayers = 1`, then call `encoder:forward` multiple times manually.

**In-place operations** — `fill`, `zero_`, and direct writes to `t.data` are not tracked by autograd. Use them only on fresh tensors or after detaching.
