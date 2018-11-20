function [net,info]=learn_alexnet(model,Name_batch)
load(['./data/config/',Name_batch,'/conf.mat'],'conf');
conf.data.Name_batch=Name_batch;
opts.dataDir=conf.data.dataDir;
opts.labelDir=conf.data.labelDir;
opts.expDir=fullfile(conf.output.dir,conf.data.Name_batch);
opts.imdbPath=fullfile(opts.expDir,'imdb.mat');
opts.whitenData=true;
opts.contrastNormalization=true;
opts.networkType='simplenn';
try
    gpuDevice(2);
    opts.train=struct('gpus',2);
catch
    error('Errors here: GPU invalid.\n')
end

%% Prepare model
net=alexnet_init();


%% Prepare data
if exist(opts.imdbPath,'file')
  imdb=load(opts.imdbPath) ;
else
    IsTrain = true;
  imdb=getImdb(opts.dataDir,opts.labelDir, IsTrain);
  if ~exist(opts.expDir, 'dir')
      mkdir(opts.expDir);
  end
  save(opts.imdbPath,'-struct','imdb','-v7.3');
end

net.meta.classes.name=imdb.meta.classes(:)';

%% Train
[net,info]=alexnet_train(net,imdb,getBatch(opts),'expDir',opts.expDir,net.meta.trainOpts,opts.train,'val',find(imdb.images.set==2));
end


function fn = getBatch(opts)
bopts = struct('numGpus', numel(opts.train.gpus)) ;
fn = @(x,y) getSimpleBatch(bopts,x,y) ;
end


function [images,labels]=getSimpleBatch(opts, imdb, batch)
images = imdb.images.data(:,:,:,batch) ;
labels = imdb.images.labels(:,:,:,batch) ;
labels = reshape(labels,[1,1,1,numel(batch)]);
if opts.numGpus > 0
  images = gpuArray(images) ;
end
end
