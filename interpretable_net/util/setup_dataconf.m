function setup_dataconf()

%% settings

conf.data.metaDir = '../data/images_for_texture/';
mkdir(conf.data.metaDir);
conf.data.dataDir = strcat(conf.data.metaDir,'data/');
conf.data.textureNetDir = '../texture_net/data/texture';

conf.data.minArea=50^2;

conf.output.dir='./data/';
mkdir([conf.output.dir,'config/']);
save([conf.output.dir,'config/','/conf.mat'],'conf');

addpath(genpath('./tool'));