function [net,info]=learn_vgg(model,Name_batch)
load(['./mat/mat/',Name_batch,'/conf.mat'],'conf');%mat/
conf.data.Name_batch=Name_batch;
opts.dataDir=conf.data.imgdir;
opts.expDir=fullfile(conf.output.dir,conf.data.Name_batch);
opts.textureNetDir=conf.data.textureNetDir;
opts.imdbPath=fullfile(opts.expDir,'imdb.mat');
opts.whitenData=true;
opts.contrastNormalization=true;
opts.networkType='simplenn';
try
    gpuID = 3;
    gpuDevice(gpuID);
    opts.train=struct('gpus',gpuID);
catch
    error('Errors here: GPU invalid.\n')
end

%% Prepare model
[net, texture_net]=my_vgg_init(opts.textureNetDir);

%% Prepare data
if exist(opts.imdbPath,'file')
  imdb=load(opts.imdbPath) ;
else
  IsTrain = true;
  imdb=getImdb(conf.data.Name_batch,conf,net.meta,IsTrain);
  if ~exist(opts.expDir, 'dir')
      mkdir(opts.expDir);
  end
  save(opts.imdbPath,'-struct','imdb');
end

net.meta.classes.name=imdb.meta.classes(:)';


%% Train
warning('off','all');
[net,info]=my_vgg_train(net,texture_net, imdb,getBatch(opts),'expDir',opts.expDir,net.meta.trainOpts,opts.train,'val',find(imdb.images.set==2));
end


function fn = getBatch(opts)
bopts = struct('numGpus', numel(opts.train.gpus)) ;
fn = @(x,y) getSimpleBatch(bopts,x,y) ;
end


function [images,labels]=getSimpleBatch(opts, imdb, batch)
images = imdb.images.data(:,:,:,batch) ;
labels = reshape(imdb.images.labels(:,batch),[1,1,size(imdb.images.labels,1),numel(batch)]);
if opts.numGpus > 0
  images = gpuArray(images) ;
end
end
