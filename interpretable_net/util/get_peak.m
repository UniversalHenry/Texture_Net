function [peaks, peak_coord] = get_peak(x)
%x is [height, width, channel, batch]
%masks is [4, channel, batch]

s = size(x);
%% its now only 1 peak
peaks = gpuArray(zeros(1,1,s(3),s(4),'single'));
peak_coord = gpuArray(zeros(2,s(3),s(4),'uint16'));

%peak and its coordinates
[tmpmax1, ind1] = max(x,[],1);
[max1, ind2] = max(tmpmax1,[],2);
ind1 = squeeze(ind1);
ind2 = squeeze(ind2);
tmp = ind2 + repmat((size(ind1, 1)*(0:s(3)-1))', [1, s(4)]);
tmp = tmp + repmat(size(ind1, 1)*s(3)*(0:s(4)-1), [s(3), 1]);
ind1 = ind1(tmp);
peaks(1,1,:,:) = max1;
peak_coord(1,:,:) = ind1;
peak_coord(2,:,:) = ind2;

%%
% ind1_re = reshape(ind1, [1,1,s(3)*s(4)]);
% ind2_re = reshape(ind2, [1,1,s(3)*s(4)]);
% ind1_re2 = repmat(ind1_re, [s(1), s(2), 1]);
% ind2_re2 = repmat(ind2_re, [s(1), s(2), 1]);
% 
% [hh,ww] = ndgrid((1:s(1)),(1:s(2)), (1:s(4)*s(3)));
% hh = hh - ind1_re2;
% ww = ww - ind2_re2;
% m = single((hh.^2+ww.^2)>=min_sep.^2);
% m = reshape(m, s);
% x(m==0) = -Inf;
% 
% % second peak
% [tmpmax2, ind3] = max(x,[],1);
% [max2, ind4] = max(tmpmax2,[],2);
% ind3 = squeeze(ind3);
% ind4 = squeeze(ind4);
% tmp2 = ind4 + repmat((size(ind3, 1)*(0:s(3)-1))', [1, s(4)]);
% tmp2 = tmp2 + repmat(size(ind3, 1)*s(3)*(0:s(4)-1), [s(3), 1]);
% ind3 = ind3(tmp2);
% peaks(1,2,:,:) = max2;
% peak_coord(3,:,:) = ind3;
% peak_coord(4,:,:) = ind4;
% 
% 
% % for masks
% [m_ind3, m_ind4] = get_coord(ind3,ind4,[s(1),s(2)],[13,13]);
% masks(3,:,:) = m_ind3;
% masks(4,:,:) = m_ind4;
end


