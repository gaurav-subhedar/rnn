
require 'nngraph'
require 'optim'
require 'lfs'
require 'util.OneHot'
require 'util.misc'
require 'torch'
require 'nn'



local CharSplitLMMinibatchLoader = require 'util.CharSplitLMMinibatchLoader'
local model_utils = require 'util.model_utils'

local LSTM = require 'model.LSTM'
local RNN = require 'model.RNN'

cmd = torch.CmdLine()
cmd:text()
cmd:text('Training model')
cmd:text()
cmd:text('Options')
-- data
cmd:option('-data_dir','data','data dir. Should contain file input.txt with input data')
-- model params
cmd:option('-rnn_size', 512, 'size of LSTM internal state')
cmd:option('-num_layers', 3, 'number of layers in the LSTM')
cmd:option('-model', 'lstm', 'lstm')
-- optimization
cmd:option('-learning_rate',2e-3,'learning rate')
cmd:option('-learning_rate_decay',0.97,'learning rate decay')
cmd:option('-learning_rate_decay_delay',10,'in number of epochs, when to start decaying the learning rate')
cmd:option('-decay_rate',0.95,'decay rate for rmsprop')
cmd:option('-dropout',0,'dropout for regularization')
cmd:option('-seq_length',8,'number of timesteps to unroll for')
cmd:option('-batch_size',50,'number of sequences to train on in parallel')
cmd:option('-max_epochs',20,'number of full passes through the training data')
cmd:option('-gradient_clip',5,'clip gradients at this value')
cmd:option('-train_data_frac',0.95,'fraction of data that goes into train set')
cmd:option('-val_data_frac',0.05,'fraction of data that goes into validation set')
           
-- bookkeeping
cmd:option('-seed',123,'torch manual random number generator seed')
cmd:option('-no_steps_loss',1,'number of steps for printing the loss')
cmd:option('-ite_eval_validation',1000,'iterations for evaluation on validation data')
cmd:option('-model_dir', 'models', 'dir for models')
cmd:option('-model_name','lstm','filename for model. Will be inside model_dir/')
-- GPU/CPU
cmd:option('-gpuid',0,'which gpu to use. -1 is no GPU')
cmd:text()

-- parse input params
opt = cmd:parse(arg)
torch.manualSeed(opt.seed)
-- train / val / test split for data, in fractions
local test_frac = math.max(0, 1 - (opt.train_data_frac + opt.val_data_frac))
local split_sizes = {opt.train_data_frac, opt.val_data_frac, test_frac} 

-- initialize cunn/cutorch for training on the GPU
if opt.gpuid >= 0 then
    local ok, cunn = pcall(require, 'cunn')
    local ok2, cutorch = pcall(require, 'cutorch')
    if not ok then print('package cunn not found!') end
    if not ok2 then print('package cutorch not found!') end
    if ok and ok2 then
        print('using CUDA on GPU ' .. opt.gpuid .. '...')
        cutorch.setDevice(opt.gpuid + 1) -- note +1 to make it 0 indexed
        cutorch.manualSeed(opt.seed)
    else
        print('Falling back on CPU mode')
        opt.gpuid = -1 -- overwrite user setting
    end
end

-- create the data loader class
local loader = CharSplitLMMinibatchLoader.create(opt.data_dir, opt.batch_size, opt.seq_length, split_sizes)
-- the number of distinct characters
local vocab_size = loader.vocab_size  
local vocab = loader.vocab_mapping
print('vocab size: ' .. vocab_size)
-- make sure output directory exists
if not path.exists(opt.model_dir) then lfs.mkdir(opt.model_dir) end

-- define the model

print('creating an ' .. opt.model .. ' with ' .. opt.num_layers .. ' layers')
protos = {}
if opt.model == 'lstm' then
    protos.rnn = LSTM.lstm(vocab_size, opt.rnn_size, opt.num_layers, opt.dropout)
elseif opt.model == 'rnn' then
    protos.rnn = RNN.rnn(vocab_size, opt.rnn_size, opt.num_layers, opt.dropout)
end
protos.criterion = nn.ClassNLLCriterion()

-- the initial state of the hidden states
init_state = {}
for L=1,opt.num_layers do
    local h_init = torch.zeros(opt.batch_size, opt.rnn_size)
    if opt.gpuid >=0 then h_init = h_init:cuda() end    
    table.insert(init_state, h_init:clone())
    if opt.model == 'lstm' then
        table.insert(init_state, h_init:clone())
    end
end

-- ship the model to the GPU if desired
if opt.gpuid >= 0 then
    for k,v in pairs(protos) do v:cuda() end
end

-- put the above things into one flattened parameters tensor
params, grad_params = model_utils.combine_all_parameters(protos.rnn)

-- initialization
params:uniform(-0.08, 0.08)

-- initialize the LSTM forget gates with slightly higher biases to encourage remembering in the beginning
if opt.model == 'lstm' then
    for layer_idx = 1, opt.num_layers do
        for _,node in ipairs(protos.rnn.forwardnodes) do
            if node.data.annotations.name == "i2h_" .. layer_idx then
                print('setting forget gate biases to 1 in LSTM layer ' .. layer_idx)
                -- the gates are, in order, i,f,o,g, so f is the 2nd block of weights
                node.data.module.bias[{{opt.rnn_size+1, 2*opt.rnn_size}}]:fill(1.0)
            end
        end
    end
end

print('number of parameters in the model: ' .. params:nElement())
-- make clones after flattening, reallocate memory
clones = {}
for name,proto in pairs(protos) do
    print('cloning ' .. name)
    clones[name] = model_utils.clone_many_times(proto, opt.seq_length, not proto.parameters)
end

-- preprocessing helper function
function prepro(x,y)
    -- swap the axes for faster indexing
    x = x:transpose(1,2):contiguous() 
    y = y:transpose(1,2):contiguous()
    if opt.gpuid >= 0 then
        x = x:float():cuda()
        y = y:float():cuda()
    end
    return x,y
end

-- evaluate the loss
function eval_split(split_index, max_batches)
    print('evaluating loss over split index ' .. split_index)
    local n = loader.split_sizes[split_index]
    if max_batches ~= nil then n = math.min(max_batches, n) end

    -- move batch iteration pointer for this split to front
    loader:reset_batch_pointer(split_index) 
    local loss = 0
    local rnn_state = {[0] = init_state}
    
    -- iterate over batches in the split
    for i = 1,n do 
        -- get a batch
        local x, y = loader:next_batch(split_index)
        x,y = prepro(x,y)
        -- forward pass
        for t=1,opt.seq_length do
            clones.rnn[t]:evaluate()
            local lst = clones.rnn[t]:forward{x[t], unpack(rnn_state[t-1])}
            rnn_state[t] = {}
            for i=1,#init_state do table.insert(rnn_state[t], lst[i]) end
            prediction = lst[#lst] 
            loss = loss + clones.criterion[t]:forward(prediction, y[t])
        end        
        rnn_state[0] = rnn_state[#rnn_state]
        print(i .. '/' .. n .. '...')
    end

    loss = loss / opt.seq_length / n
    return loss
end

-- do forward bakcward pass
local init_state_global = clone_list(init_state)
function feval(x)
    if x ~= params then
        params:copy(x)
    end
    grad_params:zero()

    
    local x, y = loader:next_batch(1)
    x,y = prepro(x,y)
    -- forward pass 
    local rnn_state = {[0] = init_state_global}
    local predictions = {}
    local loss = 0
    for t=1,opt.seq_length do
        clones.rnn[t]:training()
        local lst = clones.rnn[t]:forward{x[t], unpack(rnn_state[t-1])}
        rnn_state[t] = {}
        for i=1,#init_state do table.insert(rnn_state[t], lst[i]) end 
        -- extract the state, without output, last element is the prediction
        predictions[t] = lst[#lst]
        loss = loss + clones.criterion[t]:forward(predictions[t], y[t])
    end
    loss = loss / opt.seq_length
    --backward pass
    -- initialize gradient at time t to be 0
    local drnn_state = {[opt.seq_length] = clone_list(init_state, true)}
    for t=opt.seq_length,1,-1 do
        -- backpropagate through loss and softmax
        local doutput_t = clones.criterion[t]:backward(predictions[t], y[t])
        table.insert(drnn_state[t], doutput_t)
        local dlst = clones.rnn[t]:backward({x[t], unpack(rnn_state[t-1])}, drnn_state[t])
        drnn_state[t-1] = {}
        for k,v in pairs(dlst) do
            if k > 1 then 
                drnn_state[t-1][k-1] = v
            end
        end
    end
   
    init_state_global = rnn_state[#rnn_state]
    grad_params:clamp(-opt.gradient_clip, opt.gradient_clip)
    return loss, grad_params
end

-- optimization
train_losses = {}
val_losses = {}
local optim_state = {learningRate = opt.learning_rate, alpha = opt.decay_rate}
local iterations = opt.max_epochs * loader.ntrain
local iterations_per_epoch = loader.ntrain
local loss0 = nil
for i = 1, iterations do
    local epoch = i / loader.ntrain

    local timer = torch.Timer()
    local _, loss = optim.rmsprop(feval, params, optim_state)
    local time = timer:time().real
    
    local train_loss = loss[1]
    train_losses[i] = train_loss

    -- exponential learning rate decay
    if i % loader.ntrain == 0 and opt.learning_rate_decay < 1 then
        if epoch >= opt.learning_rate_decay_delay then
            local decay_factor = opt.learning_rate_decay
            optim_state.learningRate = optim_state.learningRate * decay_factor
            print('decayed learning rate by a factor ' .. decay_factor .. ' to ' .. optim_state.learningRate)
        end
    end

    
    if i % opt.ite_eval_validation == 0 or i == iterations then
        -- evaluate loss on validation data
        local val_loss = eval_split(2)
        val_losses[i] = val_loss

        local model_name = string.format('%s/lm_%s_epoch%.2f_%.4f.t7', opt.model_dir, opt.model_name, epoch, val_loss)
        print('saving checkpoint to ' .. model_name)
        local checkpoint = {}
        checkpoint.protos = protos
        checkpoint.opt = opt
        checkpoint.train_losses = train_losses
        checkpoint.val_loss = val_loss
        checkpoint.val_losses = val_losses
        checkpoint.i = i
        checkpoint.epoch = epoch
        checkpoint.vocab = loader.vocab_mapping
        torch.save(model_name, checkpoint)
    end

    if i % opt.no_steps_loss == 0 then
        print(string.format("%d/%d (epoch %.3f), train_loss = %6.8f, grad/param norm = %6.4e, time/batch = %.4fs", i, iterations, epoch, train_loss, grad_params:norm() / params:norm(), time))
    end
   
    if i % 10 == 0 then collectgarbage() end

    -- handle early stopping if things are going really bad
    if loss[1] ~= loss[1] then
        print('loss is NaN.')
        break -- halt
    end
    if loss0 == nil then loss0 = loss[1] end
    if loss[1] > loss0 * 3 then
        print('loss is exploding, aborting.')
        break -- halt
    end
end


