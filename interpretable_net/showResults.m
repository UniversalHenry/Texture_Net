function showResults(Name_batch)
load(['./mat/',Name_batch,'/conf.mat'],'conf');
addpath(genpath('./util'));

% net for evaluation
expDir=fullfile(conf.output.dir,Name_batch);
modelPath = @(ep) fullfile(expDir, sprintf('net-epoch-%d.mat', ep));
epochNum=findLastCheckpoint(expDir);
[net, ~, ~] = loadState(modelPath(epochNum));
side = net.layers{end-1}.side;
net.layers = net.layers(1:30);
net.layers{end+1} = struct('type', 'eval', 'name', 'evaluation',...
                            'side',side) ;

% texture_net
modelPath = @(ep) fullfile(conf.data.textureNetDir, sprintf('net-epoch-%d.mat', ep));
start = findLastCheckpoint(conf.data.textureNetDir) ;
[texture_net, ~, ~] = loadState(modelPath(start));
texture_net.layers(end) = [];

IsTrain = false;
imdb=getImdb(Name_batch,conf,net.meta,IsTrain);

resultDir = fullfile(expDir, 'results');
if ~exist(resultDir, 'dir')
  mkdir(resultDir);
end

getResult(net, texture_net, imdb, resultDir);

end

function getResult(net, texture_net, imdb, expDir)
im = gpuArray(imresize(imdb.images.data,[224,224])) ;
filter_num = size(net.layers{end-2}.weights{1},4);
s=size(im);

score = zeros(filter_num,0);
peaks = zeros(filter_num,0);
gram = zeros(filter_num,filter_num,0);

evalMode = 'test';
dzdy = [] ;
res = [] ;
net =vl_simplenn_move(net, 'gpu') ;

scoreDir = fullfile(expDir, 'score');
if ~exist(scoreDir, 'dir')
  mkdir(scoreDir);
end

peakDir = fullfile(expDir, 'peak');
if ~exist(peakDir, 'dir')
  mkdir(peakDir);
end

gramDir = fullfile(expDir, 'gram');
if ~exist(gramDir, 'dir')
  mkdir(gramDir);
end

bs = 6;
% split into batches
for t=1:bs:s(4)
    fprintf('Batch %d/%d\n',fix((t-1)/bs)+1, ceil(s(4)/bs));
    im_b = im(:,:,:,t:min(t+bs-1,s(4)));
    res = my_vgg(net, texture_net, im_b, [], dzdy, res, ...
                      [], [], [],...
                      [],...
                      'accumulate', false, ...
                      'mode', evalMode, ...
                      'conserveMemory',true, ...
                      'backPropDepth', +inf, ...
                      'sync', false, ...
                      'cudnn', true, ...
                      'holdOn', false) ;

    score = cat(2,score,gather(res(end).score));
    peaks = cat(2,peaks,gather(res(end).peaks));
    gram = cat(3,gram,gather(res(end).gram_matrix));
%     save([scoreDir,sprintf('/batch_%d.mat',fix((t-1)/bs)+1)],'score');
%     save([peakDir,sprintf('/batch_%d.mat',fix((t-1)/bs)+1)],'peaks');
%     save([gramDir,sprintf('/batch_%d.mat',fix((t-1)/bs)+1)],'gram_matrix');
end

% scores_ = dir([scoreDir,'/*.mat']);
% peaks_ = dir([peakDir,'/*.mat']);
% gram_ = dir([gramDir,'/*.mat']);
% 
% 
% score = zeros(filter_num,0);
% for scr=scores_'
%     s = load(fullfile(scoreDir, scr.name));
%     score = cat(2,score,s.score);
% end
% 
% peaks = zeros(filter_num,0);
% for pk=peaks_'
%     s = load(fullfile(peakDir, pk.name));
%     peaks = cat(2,peaks,s.peaks);
% end
% 
% gram = zeros(filter_num,filter_num,0);
% for g=gram_'
%     s = load(fullfile(gramDir, g.name));
%     gram = cat(3,gram,s.gram_matrix);
% end

for k=1:filter_num
    fprintf('Saving to score distribution.\n');
    f=figure('visible', 'off');
    hist(score(k,:), 30);
    hgsave(f,[expDir,sprintf('/score_dist_filter_%d.fig',k)]);
    fprintf('Saving to peak distribution.\n');
    f2=figure('visible', 'off');
    hist(peaks(k,:), 30);
    hgsave(f2,[expDir,sprintf('/peak_dist_filter_%d.fig',k)]);
end

save(fullfile(expDir,'gram_matrix.mat'),'gram');
fprintf('Saved gram matrix\n');
              
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