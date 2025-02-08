# NN_for_miniworld

使用示例:
\`

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

\`
