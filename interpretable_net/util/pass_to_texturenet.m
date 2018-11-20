function [labels, peaks, peak_coord] = pass_to_texturenet(feature_map, class, texture_net, x, side, opts)
%% pass to texture net
addpath(genpath('../../texture_net/'));
[peaks, peak_coord] = get_peak(feature_map);
% allocate for labels
labels = gpuArray(zeros(1,1,size(feature_map,3),size(feature_map,4)));

res_ = [];
if strcmp(opts.mode, 'train')
  dzdy_ = 1 ;
  evalMode = 'normal' ;
else
  dzdy_ = [] ;
  evalMode = 'test' ;
end

texture_net =vl_simplenn_move(texture_net, 'gpu') ;
gramnet = load('imagenet-vgg-f.mat');
gramnet = vl_simplenn_tidy(gramnet) ;
gramnet = our_vl_simplenn_move(gramnet, 'gpu') ;

class=class>=0;

% if ~exist('tmpfile')
%     tmpfile = 0;
%     save tmpfile tmpfile;
% end
% load tmpfile;

for i = 1:size(class,4)/2
    if (class(:,:,:,i*2-1)*class(:,:,:,i*2))==0
        labels(:,:,:,i*2-1)=-2;
        labels(:,:,:,i*2)=-2;
    else
        x_ = get_peak_area(x(:,:,:,i*2-1:i*2) ,feature_map(:,:,:,i*2-1:i*2), ...
             peaks(:,:,:,i*2-1:i*2), peak_coord(:,:,i*2-1:i*2), side,texture_net.meta.normalization.imageSize);
        x_= single(x_);
        gramlabels=get_gram_score(x_(:,:,:,1:(size(x_,4)/2)),gramnet);
        res_=my_alexnet_forward(texture_net, x_, dzdy_, res_, ...
                              'accumulate', opts.accumulate, ...
                              'mode', evalMode, ...
                              'conserveMemory', opts.conserveMemory, ...
                              'backPropDepth', opts.backPropDepth, ...
                              'sync', opts.sync, ...
                              'cudnn', opts.cudnn, ...
                              'parameterServer', opts.parameterServer, ...
                              'holdOn', opts.holdOn) ;
        texturelabels = reshape((res_(end).x(:,:,:,1:size(x_,4)/2)+res_(end).x(:,:,:,(size(x_,4)/2+1):end))./2,...
            [1,1,size(x_,4)/2]);
        labels(:,:,:,i*2-1) = 0.5*gramlabels + 0.5*texturelabels;
        labels(:,:,:,i*2) = labels(:,:,:,i*2-1);
%         %% show imageds compared
%         labeltmp=gather(reshape(labels(:,:,:,i*2),1,1,1,[]));
%         for j = 1:numel(labeltmp)
%             tmpim=gather(uint8(x_(:,:,1:3,j)+128));
%             tmpim = insertText(tmpim,[1,50],labeltmp(:,:,:,j),'AnchorPoint','LeftBottom');
%             imwrite(tmpim, fullfile('./tmp/', sprintf('b%d_f%d_%dp1.jpg',i*2-1,j,tmpfile)));
%             tmpim=gather(uint8(x_(:,:,4:6,j)+128));
%             tmpim = insertText(tmpim,[1,50],labeltmp(:,:,:,j),'AnchorPoint','LeftBottom');
%             imwrite(tmpim, fullfile('./tmp/', sprintf('b%d_f%d_%dp2.jpg',i*2,j,tmpfile)));
%         end
    end
end
end


