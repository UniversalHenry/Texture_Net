function setup_dataconf()

%% settings
conf.data.metaDir = '../data/texture/';
conf.data.dataDir = strcat(conf.data.metaDir,'data/');
conf.data.labelDir = strcat(conf.data.metaDir, 'label/');
conf.data.readCode='./data/data_input/';

conf.data.minArea=50^2;

conf.output.dir='./data/';

mkdir([conf.output.dir,'config/','texture']);
save([conf.output.dir,'config/','texture','/conf.mat'],'conf');

addpath(genpath('./tool'));
addpath(conf.data.readCode);