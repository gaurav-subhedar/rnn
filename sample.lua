
require 'torch'
require 'nn'
require 'nngraph'
require 'optim'
require 'lfs'

require 'util.OneHot'
require 'util.misc'

cmd = torch.CmdLine()
cmd:text()
cmd:text('Generate next character from model')
cmd:text()
cmd:text('Options')
-- required:
cmd:argument('-model','model to use')
-- optional parameters
cmd:option('-seed',123,'random number generator\'s seed')
cmd:option('-sample',1,' 0 to use max at each timestep, 1 to sample at each timestep')
cmd:option('-prefix',"ann",'used as a prompt to "seed" the state of the LSTM using a given sequence, before we sample.')
cmd:option('-length',1,'number of characters to sample')
cmd:option('-temperature',1,'temperature of sampling')
cmd:option('-gpuid',0,'which gpu to use. -1 = use CPU')
cmd:option('-verbose',1,'set to 0 to ONLY print the sampled text, no diagnostics')
cmd:text()

-- parse input params
opt = cmd:parse(arg)

-- utility function wrapping a print
function gprint(str)
    if opt.verbose == 1 then print(str) end
end

-- check for cunn/cutorch if user wants to use the GPU
if opt.gpuid >= 0 then
    local ok, cunn = pcall(require, 'cunn')
    local ok2, cutorch = pcall(require, 'cutorch')
    if not ok then gprint('package cunn not found!') end
    if not ok2 then gprint('package cutorch not found!') end
    if ok and ok2 then
        cutorch.setDevice(opt.gpuid + 1) -- +1 to make it 0 indexed
        cutorch.manualSeed(opt.seed)
    else
        gprint('Falling back on CPU mode')
        opt.gpuid = -1 -- overwrite user setting
    end
end


torch.manualSeed(opt.seed)

-- load the model
if not lfs.attributes(opt.model, 'mode') then
    gprint('Error: File ' .. opt.model .. ' does not exist.')
end
cur_model = torch.load(opt.model)
protos = cur_model.protos

-- put in eval mode so that dropout works properly
protos.rnn:evaluate() 

-- initialize the vocab
local vocab = cur_model.vocab
local rev_vocab = {}
for c,i in pairs(vocab) do rev_vocab[i] = c end

-- initialize rnn state to all zeros
local cur_state
cur_state = {}

for L = 1,cur_model.opt.num_layers do
    -- c and h for all layers
    local h_init = torch.zeros(1, cur_model.opt.rnn_size):double()
    if opt.gpuid >= 0 then h_init = h_init:cuda() end
    table.insert(cur_state, h_init:clone())
    if cur_model.opt.model == 'lstm' then
        table.insert(cur_state, h_init:clone())
    end
end
state_size = #cur_state

-- few seeded steps
local seed_text = opt.prefix
if string.len(seed_text) > 0 then
    for c in seed_text:gmatch'.' do
        prev_char = torch.Tensor{vocab[c]}
        io.write(rev_vocab[prev_char[1]])
        if opt.gpuid >= 0 and opt.opencl == 0 then prev_char = prev_char:cuda() end
        local lst = protos.rnn:forward{prev_char, unpack(cur_state)}
        -- lst is a list of states+output. Get everything but last piece
        cur_state = {}
        for i=1,state_size do table.insert(cur_state, lst[i]) end
        -- last element holds the log probabilities
        prediction = lst[#lst]
    end
else
    -- fill with uniform probabilities over characters
    prediction = torch.Tensor(1, #rev_vocab):fill(1)/(#rev_vocab)
    if opt.gpuid >= 0 and opt.opencl == 0 then prediction = prediction:cuda() end    
end

local retval = ""

-- start sampling

for i=1, opt.length do

    -- log probabilities from the previous timestep
    if opt.sample == 0 then
        -- use argmax
        local _, prev_char_ = prediction:max(2)
        prev_char = prev_char_:resize(1)
    else
        -- use sampling
        prediction:div(opt.temperature)
        local probs = torch.exp(prediction):squeeze()
        probs:div(torch.sum(probs)) 
        --gprint('now printing probs------')
        --io.write(tostring(probs))
        prev_char = torch.multinomial(probs:float(), 1):resize(1):float()
    end

    -- forward the rnn for next character
    local lst = protos.rnn:forward{prev_char, unpack(cur_state)}
    cur_state = {}
    for i=1,state_size do table.insert(cur_state, lst[i]) end
    -- last element holds the log probabilities
    prediction = lst[#lst] 
    for xa, xb in pairs( lst ) do
   	--print(xa, xb)
    end
    --gprint('now printing prediction------')
    --io.write(tostring(prediction))
    --gprint('now printing prev_char------')
    --io.write(tostring(prev_char))
    io.write(rev_vocab[prev_char[1]])
    retval = retval .. rev_vocab[prev_char[1]]
end
--io.write('\n')
--io.write(retval) 
io.flush()
return retval

