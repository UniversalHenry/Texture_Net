function showResults(Name_batch)
load(['./data/config/',Name_batch,'/conf.mat'],'conf');
addpath(genpath('../code/tool'));
addpath(genpath('../code/tool/edges-master/piotr_toolbox/external'));
toolboxCompile;

%% compute classification error and location stability
fileRoot='./data';

epochNum=50;
binaryerror=getResult({Name_batch},fileRoot,epochNum);
fprintf('binary error %f     location stability %f\n',binaryerror)


function binaryerror = getResult(nameList,fileRoot,epochNum)
num=numel(nameList);
binaryerror=zeros(num,1);
for i=1:num
    filename=sprintf('./%s/%s/net-epoch-%d.mat',fileRoot,nameList{i},epochNum);
    try
        net=load(filename);
        binaryerror(i)=net.stats.val(end).binerr;
        clear net
    catch
        continue;
    end
end


