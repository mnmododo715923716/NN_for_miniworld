local Matrix = {}

-- 创建矩阵构造函数
function Matrix:new(rows, cols, data)
    local m = {
        rows = rows,
        cols = cols,
        data = data or {}
    }
    setmetatable(m, self)
    self.__index = self
    return m
end

--[[ 基本矩阵操作 ]]--

-- 生成零矩阵
function Matrix.zero(rows, cols)
    local data = {}
    for i = 1, rows do
        data[i] = {}
        for j = 1, cols do
            data[i][j] = 0
        end
    end
    return Matrix:new(rows, cols, data)
end

-- 生成单位矩阵
function Matrix.identity(n)
    local data = {}
    for i = 1, n do
        data[i] = {}
        for j = 1, n do
            data[i][j] = (i == j) and 1 or 0
        end
    end
    return Matrix:new(n, n, data)
end

-- 矩阵加法
function Matrix.add(a, b)
    if a.rows ~= b.rows or a.cols ~= b.cols then
        error("矩阵维度不匹配")
    end
    local result = Matrix.zero(a.rows, a.cols)
    for i = 1, a.rows do
        for j = 1, a.cols do
            result.data[i][j] = a.data[i][j] + b.data[i][j]
        end
    end
    return result
end

-- 矩阵减法
function Matrix.sub(a, b)
    if a.rows ~= b.rows or a.cols ~= b.cols then
        error("矩阵维度不匹配")
    end
    local result = Matrix.zero(a.rows, a.cols)
    for i = 1, a.rows do
        for j = 1, a.cols do
            result.data[i][j] = a.data[i][j] - b.data[i][j]
        end
    end
    return result
end

-- 标量乘法
function Matrix.scalar_mul(m, scalar)
    local result = Matrix.zero(m.rows, m.cols)
    for i = 1, m.rows do
        for j = 1, m.cols do
            result.data[i][j] = m.data[i][j] * scalar
        end
    end
    return result
end

-- 矩阵乘法
function Matrix.mul(a, b)
    if a.cols ~= b.rows then
        error("矩阵维度不匹配")
    end
    local result = Matrix.zero(a.rows, b.cols)
    for i = 1, a.rows do
        for j = 1, b.cols do
            local sum = 0
            for k = 1, a.cols do
                sum = sum + a.data[i][k] * b.data[k][j]
            end
            result.data[i][j] = sum
        end
    end
    return result
end

-- 转置矩阵
function Matrix.transpose(m)
    local result = Matrix.zero(m.cols, m.rows)
    for i = 1, m.rows do
        for j = 1, m.cols do
            result.data[j][i] = m.data[i][j]
        end
    end
    return result
end

--[[ 高级矩阵操作 ]]--

-- 计算行列式（递归方法）
function Matrix.determinant(m)
    if m.rows ~= m.cols then
        error("非方阵无法计算行列式")
    end
    if m.rows == 1 then
        return m.data[1][1]
    end
    if m.rows == 2 then
        return m.data[1][1]*m.data[2][2] - m.data[1][2]*m.data[2][1]
    end
    
    local det = 0
    for j = 1, m.cols do
        local minor = Matrix.zero(m.rows-1, m.cols-1)
        for i = 2, m.rows do
            local col = 1
            for k = 1, m.cols do
                if k ~= j then
                    minor.data[i-1][col] = m.data[i][k]
                    col = col + 1
                end
            end
        end
        det = det + (m.data[1][j] * ((-1)^(1+j)) * Matrix.determinant(minor))
    end
    return det
end

-- 高斯-约旦消元求逆
function Matrix.inverse(m)
    if m.rows ~= m.cols then
        error("非方阵不可逆")
    end
    local n = m.rows
    local aug = Matrix.zero(n, 2*n)
    
    -- 构造增广矩阵 [A|I]
    for i = 1, n do
        for j = 1, n do
            aug.data[i][j] = m.data[i][j]
        end
        aug.data[i][i+n] = 1
    end
    
    -- 前向消元
    for k = 1, n do
        -- 寻找主元
        local max_row = k
        for i = k+1, n do
            if math.abs(aug.data[i][k]) > math.abs(aug.data[max_row][k]) then
                max_row = i
            end
        end
        
        -- 交换行
        aug.data[k], aug.data[max_row] = aug.data[max_row], aug.data[k]
        
        local pivot = aug.data[k][k]
        if math.abs(pivot) < 1e-10 then
            error("矩阵不可逆")
        end
        
        -- 归一化主元行
        for j = k, 2*n do
            aug.data[k][j] = aug.data[k][j] / pivot
        end
        
        -- 消去其他行
        for i = 1, n do
            if i ~= k then
                local factor = aug.data[i][k]
                for j = k, 2*n do
                    aug.data[i][j] = aug.data[i][j] - factor * aug.data[k][j]
                end
            end
        end
    end
    
    -- 提取逆矩阵
    local inv = Matrix.zero(n, n)
    for i = 1, n do
        for j = 1, n do
            inv.data[i][j] = aug.data[i][j+n]
        end
    end
    return inv
end

-- LU分解
function Matrix.lu_decomposition(m)
    if m.rows ~= m.cols then
        error("需要方阵进行LU分解")
    end
    local n = m.rows
    local L = Matrix.identity(n)
    local U = Matrix.zero(n, n)
    
    for k = 1, n do
        U.data[k][k] = m.data[k][k]
        for j = k, n do
            local sum = 0
            for s = 1, k-1 do
                sum = sum + L.data[k][s] * U.data[s][j]
            end
            U.data[k][j] = m.data[k][j] - sum
        end
        
        for i = k+1, n do
            local sum = 0
            for s = 1, k-1 do
                sum = sum + L.data[i][s] * U.data[s][k]
            end
            L.data[i][k] = (m.data[i][k] - sum) / U.data[k][k]
        end
    end
    return L, U
end

--[[ 实用功能 ]]--

-- 矩阵打印
function Matrix:print()
    for i = 1, self.rows do
        local line = ""
        for j = 1, self.cols do
            line = line .. string.format("%8.4f ", self.data[i][j])
        end
        print(line)
    end
end

-- 生成随机矩阵
function Matrix.random(rows, cols, min, max)
    local data = {}
    for i = 1, rows do
        data[i] = {}
        for j = 1, cols do
            data[i][j] = min + math.random() * (max - min)
        end
    end
    return Matrix:new(rows, cols, data)
end

--[[示例用法
local A = Matrix.random(3, 3, -1, 1)
print("原始矩阵 A:")
A:print()

local B = Matrix.inverse(A)
print("\n逆矩阵 A⁻¹:")
B:print()

local I = Matrix.mul(A, B)
print("\n验证 A * A⁻¹ = I:")
I:print()

local L, U = Matrix.lu_decomposition(A)
print("\nLU分解的 L 矩阵:")
L:print()
print("\nLU分解的 U 矩阵:")
U:print()
]]--



local NeuralNetwork = {
    activations = {},
    losses = {},
    optimizers = {}
}

--[[ 神经网络核心模块 ]]--
local Layer = {}
function Layer:new(units, activation)
    local layer = {
        units = units,
        activation = activation,
        weights = nil,
        bias = nil,
        input = nil,
        output = nil
    }
    setmetatable(layer, self)
    self.__index = self
    return layer
end

-- 全连接层实现
local Dense = Layer:new()
function Dense:initialize(input_shape)
    local limit = math.sqrt(6 / (input_shape + self.units))
    self.weights = Matrix.random(input_shape, self.units, -limit, limit)
    self.bias = Matrix.zero(1, self.units)
end

function Dense:forward(input)
    self.input = input
    local z = Matrix.mul(input, self.weights)
    z = Matrix.add(z, self.bias)
    self.output = NeuralNetwork.activations[self.activation](z)
    return self.output
end

--[[ 激活函数实现 ]]--
-- Sigmoid
NeuralNetwork.activations.sigmoid = function(z)
    local data = {}
    for i=1, z.rows do
        data[i] = {}
        for j=1, z.cols do
            data[i][j] = 1 / (1 + math.exp(-z.data[i][j]))
        end
    end
    return Matrix:new(z.rows, z.cols, data)
end

-- ReLU
NeuralNetwork.activations.relu = function(z)
    local data = {}
    for i=1, z.rows do
        data[i] = {}
        for j=1, z.cols do
            data[i][j] = math.max(0, z.data[i][j])
        end
    end
    return Matrix:new(z.rows, z.cols, data)
end

-- Softmax
NeuralNetwork.activations.softmax = function(z)
    local data = {}
    for i=1, z.rows do
        local row = {}
        local max_val = math.max(unpack(z.data[i]))
        local sum_exp = 0
        for j=1, z.cols do
            row[j] = math.exp(z.data[i][j] - max_val)
            sum_exp = sum_exp + row[j]
        end
        for j=1, z.cols do
            row[j] = row[j] / sum_exp
        end
        data[i] = row
    end
    return Matrix:new(z.rows, z.cols, data)
end

--[[ 损失函数 ]]--
-- 均方误差
NeuralNetwork.losses.mse = {
    forward = function(y_true, y_pred)
        local diff = Matrix.sub(y_true, y_pred)
        local squared = Matrix.scalar_mul(diff, 2)
        return Matrix.mul(diff, squared):mean()
    end,
    backward = function(y_true, y_pred)
        return Matrix.sub(y_pred, y_true)
    end
}

-- 交叉熵损失
NeuralNetwork.losses.crossentropy = {
    forward = function(y_true, y_pred)
        local eps = 1e-8
        local loss = 0
        for i=1, y_true.rows do
            for j=1, y_true.cols do
                loss = loss - y_true.data[i][j] * math.log(y_pred.data[i][j] + eps)
            end
        end
        return loss / y_true.rows
    end,
    backward = function(y_true, y_pred)
        return Matrix.sub(y_pred, y_true)
    end
}

--[[ 神经网络模型 ]]--
local Model = {}
function Model:new()
    local model = {
        layers = {},
        loss = nil,
        optimizer = nil
    }
    setmetatable(model, self)
    self.__index = self
    return model
end

function Model:add(layer)
    table.insert(self.layers, layer)
end

function Model:compile(cfg)
    self.loss = cfg.loss
    self.optimizer = cfg.optimizer
    
    -- 初始化各层权重
    local input_shape = cfg.input_shape
    for _, layer in ipairs(self.layers) do
        layer:initialize(input_shape)
        input_shape = layer.units
    end
end

function Model:forward(x)
    local output = x
    for _, layer in ipairs(self.layers) do
        output = layer:forward(output)
    end
    return output
end

function Model:backward(grad)
    -- 反向传播实现
    for i=#self.layers,1,-1 do
        local layer = self.layers[i]
        -- 计算激活函数梯度
        local activation_grad = self:_activation_grad(layer)
        grad = Matrix.hadamard(grad, activation_grad)
        
        -- 计算权重梯度
        local delta_w = Matrix.mul(Matrix.transpose(layer.input), grad)
        local delta_b = grad:sum(1)  -- 按列求和
        
        -- 更新参数
        self.optimizer:update(layer, delta_w, delta_b)
        
        -- 计算前一层梯度
        grad = Matrix.mul(grad, Matrix.transpose(layer.weights))
    end
end

function Model:train(x_train, y_train, epochs, batch_size)
    for epoch=1, epochs do
        local total_loss = 0
        for i=1, #x_train, batch_size do
            -- 获取批次数据
            local x_batch = Matrix:new(batch_size, x_train.cols, {})
            local y_batch = Matrix:new(batch_size, y_train.cols, {})
            
            -- 前向传播
            local y_pred = self:forward(x_batch)
            
            -- 计算损失
            total_loss = total_loss + self.loss.forward(y_batch, y_pred)
            
            -- 反向传播
            local grad = self.loss.backward(y_batch, y_pred)
            self:backward(grad)
        end
        print(("Epoch %d Loss: %.4f"):format(epoch, total_loss))
    end
end

--[[ 优化器实现 ]]--


-- Adam优化器
NeuralNetwork.optimizers.Adam = {
    new = function(learning_rate, beta1, beta2, epsilon)
        return {
            lr = learning_rate or 0.001,
            beta1 = beta1 or 0.9,
            beta2 = beta2 or 0.999,
            epsilon = epsilon or 1e-8,
            m = {},
            v = {},
            t = 0,
            update = function(self, layer, delta_w, delta_b)
                self.t = self.t + 1
                local lr_t = self.lr * math.sqrt(1 - self.beta2^self.t) / (1 - self.beta1^self.t)
                
                if not self.m[layer] then
                    self.m[layer] = Matrix.zero(delta_w:size())
                    self.v[layer] = Matrix.zero(delta_w:size())
                end
                
                -- 更新一阶矩估计和二阶矩估计
                self.m[layer] = Matrix.scalar_mul(self.m[layer], self.beta1) + Matrix.scalar_mul(delta_w, 1 - self.beta1)
                self.v[layer] = Matrix.scalar_mul(self.v[layer], self.beta2) + Matrix.scalar_mul(Matrix.hadamard(delta_w, delta_w), 1 - self.beta2)
                
                -- 计算偏差修正后的参数更新值
                local m_hat = Matrix.scalar_div(self.m[layer], 1 - self.beta1^self.t)
                local v_hat = Matrix.scalar_div(self.v[layer], 1 - self.beta2^self.t)
                
                -- 更新权重和偏置
                layer.weights = Matrix.sub(layer.weights, Matrix.scalar_mul(m_hat, lr_t))
                layer.bias = Matrix.sub(layer.bias, Matrix.scalar_mul(v_hat, lr_t))
            end
        }
    end
}

-- 随机梯度下降

NeuralNetwork.optimizers.SGD = {
    new = function(learning_rate)
        return {
            lr = learning_rate,
            update = function(self, layer, delta_w, delta_b)
                layer.weights = Matrix.sub(layer.weights, Matrix.scalar_mul(delta_w, self.lr))
                layer.bias = Matrix.sub(layer.bias, Matrix.scalar_mul(delta_b, self.lr))
            end
        }
    end
}

--[[ 示例：解决XOR问题 ]]--
local model = Model:new()
model:add(Dense:new(4, 'relu'))  -- 隐藏层
model:add(Dense:new(1, 'sigmoid'))  -- 输出层

model:compile{
    input_shape = 2,
    loss = NeuralNetwork.losses.mse,
    optimizer = NeuralNetwork.optimizers.SGD.new(0.1)
}

-- XOR数据集
local x_train = Matrix:new(4, 2, {
    {0, 0},
    {0, 1},
    {1, 0},
    {1, 1}
})
local y_train = Matrix:new(4, 1, {
    {0},
    {1},
    {1},
    {0}
})

-- 训练模型
model:train(x_train, y_train, 1000, 4)

-- 预测
local test_input = Matrix:new(1, 2, {{0, 1}})
local prediction = model:forward(test_input)
print("Prediction for [0, 1]:", prediction.data[1][1])