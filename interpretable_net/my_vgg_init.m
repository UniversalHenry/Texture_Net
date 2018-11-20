function [net, texture_net] = my_vgg_init(textureNetDir, varargin)
addpath(genpath('../texture_net/'));

opts.scale = 1 ;
opts.initBias = 0 ;
opts.weightDecay = 1 ;
%opts.weightInitMethod = 'xavierimproved' ;
opts.weightInitMethod = 'gaussian' ;
opts.batchNormalization = false ;
opts.networkType = 'simplenn' ;
opts.cudnnWorkspaceLimit = 1024*1024*1204 ; % 1GB
opts.classNames = {} ;
opts.classDescriptions = {} ;
opts.averageImage = zeros(3,1) ;
opts.colorDeviation = zeros(3) ;
opts = vl_argparse(opts, varargin) ;

%% vgg-16
net=load('../nets/imagenet-vgg-verydeep-16.mat');
net.layers = net.layers(1:28);
for k=1:28
    net.layers{k}.learningRate = [0 0];
    net.layers{k}.weightDecay = [0 0];
end
net = add_block(net, opts, '_texture', 3, 3, 256, 96, 1, 1) ;                     
net = add_block(net,opts, '6', 3 , 3 , 96 , 96 , 1 , 1 ) ;
net = add_pool(net,opts,'6','max', 2 , 2 , 2 , 0);
net = add_block(net,opts,'7',7,7,96,4096,1,0);
net = add_block(net,opts,'8',1,1,4096,1,1,0);
net.layers{end+1} = struct('type', 'loss', 'name', 'logistic_loss') ; 
net.layers{end+1} = struct('type', 'texture_net', 'name', 'texture_net',...
                            'side',40) ;
net.layers{end+1} = struct('type', 'myloss', 'name', 'myloss',...
                            'tau', 50) ;   
                        
%% load pretrianed texturenet
if ~exist(textureNetDir, 'dir')
    fprintf('Directory not found') ;
    exit()
end
modelPath = @(ep) fullfile(textureNetDir, sprintf('net-epoch-%d.mat', ep));
start = findLastCheckpoint(textureNetDir) ;
[texture_net, state, stats] = loadState(modelPath(start));
texture_net.layers{end} = struct('type', 'label', 'name', 'prediction') ;

%
for i=1:length(texture_net.layers)-1
    texture_net.layers{i}.learningRate = [0 0];
    texture_net.layers{i}.weightDecay = [0 0];
end

bs = 20 ;

%%
% final touches
switch lower(opts.weightInitMethod)
  case {'xavier', 'xavierimproved'}
    net.layers{end-1}.weights{1} = net.layers{end-1}.weights{1} / 10 ;
end

if ~opts.batchNormalization
  lr = logspace(-5, -6, 32) ;
else
  lr = logspace(-1, -4, 20) ;
end

net.meta.trainOpts.learningRate = lr ;
net.meta.trainOpts.numEpochs = numel(lr) ;
net.meta.trainOpts.batchSize = bs ;
net.meta.trainOpts.weightDecay = 0.0005 ;

% Fill in default values
net = vl_simplenn_tidy(net) ;

% Switch to DagNN if requested
switch lower(opts.networkType)
  case 'simplenn'
    % done
  case 'dagnn'
    net = dagnn.DagNN.fromSimpleNN(net, 'canonicalNames', true) ;
    net.addLayer('top1err', dagnn.Loss('loss', 'classerror'), ...
                 {'prediction','label'}, 'top1err') ;
    net.addLayer('top5err', dagnn.Loss('loss', 'topkerror', ...
                                       'opts', {'topK',5}), ...
                 {'prediction','label'}, 'top5err') ;
  otherwise
    assert(false) ;
end
end

% --------------------------------------------------------------------
function net = add_block(net, opts, id, h, w, in, out, stride, pad)
% --------------------------------------------------------------------
info = vl_simplenn_display(net) ;
fc = (h == info.dataSize(1,end) && w == info.dataSize(2,end)) ;
if fc
  name = 'fc' ;
else
  name = 'conv' ;
end
convOpts = {'CudnnWorkspaceLimit', opts.cudnnWorkspaceLimit} ;
net.layers{end+1} = struct('type', 'conv', 'name', sprintf('%s%s', name, id), ...
                           'weights', {{init_weight(opts, h, w, in, out, 'single'), ...
                             ones(out, 1, 'single')*opts.initBias}}, ...
                           'stride', stride, ...
                           'pad', pad, ...
                           'dilate', 1, ...
                           'learningRate', [1 2], ...
                           'weightDecay', [opts.weightDecay 0], ...
                           'opts', {convOpts}) ;
if opts.batchNormalization
  net.layers{end+1} = struct('type', 'bnorm', 'name', sprintf('bn%s',id), ...
                             'weights', {{ones(out, 1, 'single'), zeros(out, 1, 'single'), ...
                               zeros(out, 2, 'single')}}, ...
                             'epsilon', 1e-4, ...
                             'learningRate', [2 1 0.1], ...
                             'weightDecay', [0 0]) ;
end
if ~fc 
    net.layers{end+1} = struct('type', 'relu', 'name', sprintf('relu%s',id)) ;
end
if fc 
    net = add_dropout(net,opts,id);
end
end

% -------------------------------------------------------------------------
function weights = init_weight(opts, h, w, in, out, type)
% -------------------------------------------------------------------------
% See K. He, X. Zhang, S. Ren, and J. Sun. Delving deep into
% rectifiers: Surpassing human-level performance on imagenet
% classification. CoRR, (arXiv:1502.01852v1), 2015.

switch lower(opts.weightInitMethod)
  case 'gaussian'
    sc = 0.01/opts.scale ;
    weights = randn(h, w, in, out, type)*sc;
  case 'xavier'
    sc = sqrt(3/(h*w*in)) ;
    weights = (rand(h, w, in, out, type)*2 - 1)*sc ;
  case 'xavierimproved'
    sc = sqrt(2/(h*w*out)) ;
    weights = randn(h, w, in, out, type)*sc ;
  otherwise
    error('Unknown weight initialization method''%s''', opts.weightInitMethod) ;
end
end

% --------------------------------------------------------------------
function net = add_norm(net, opts, id)
% --------------------------------------------------------------------
if ~opts.batchNormalization
  net.layers{end+1} = struct('type', 'normalize', ...
                             'name', sprintf('norm%s', id), ...
                             'param', [5 1 0.0001/5 0.75]) ;
end
end

% --------------------------------------------------------------------
function net = add_dropout(net, opts, id)
% --------------------------------------------------------------------
if ~opts.batchNormalization
  net.layers{end+1} = struct('type', 'dropout', ...
                             'name', sprintf('dropout%s', id), ...
                             'rate', 0.5) ;
end
end

% --------------------------------------------------------------------
function net = add_pool(net, opts, id, method, h, w, stride, pad)
% --------------------------------------------------------------------
net.layers{end+1} = struct('type', 'pool', 'name', sprintf('pool%s', id), ...
                           'method', method, ...
                           'pool', [h w], ...
                           'stride', stride, ...
                           'pad', pad);
                       

end

% -------------------------------------------------------------------------
function epoch = findLastCheckpoint(modelDir)
% -------------------------------------------------------------------------
list = dir(fullfile(modelDir, 'net-epoch-*.mat')) ;
tokens = regexp({list.name}, 'net-epoch-([\d]+).mat', 'tokens') ;
epoch = cellfun(@(x) sscanf(x{1}{1}, '%d'), tokens) ;
epoch = max([epoch 0]) ;
end

% -------------------------------------------------------------------------
function [net, state, stats] = loadState(fileName)
% -------------------------------------------------------------------------
load(fileName, 'net', 'state', 'stats') ;
net = vl_simplenn_tidy(net) ;
if isempty(whos('stats'))
  error('Epoch ''%s'' was only partially saved. Delete this file and try again.', ...
        fileName) ;
end
end
