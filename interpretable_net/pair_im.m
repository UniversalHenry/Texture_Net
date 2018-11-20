function out = pair_im( x )
s=size(x);

idx1 = repmat(1:s(end),[s(end),1]);
idx2 = repmat((1:s(end))',[1,s(end)]);
mask = tril(true([s(end),s(end)]),-1);
idx1 = idx1(mask);
idx2 = idx2(mask);

tmp1 = x(:,:,:,:,idx1(:));
% tmp1 = repmat(x(:,:,:,id), [1,1,1,s(4)-id]);
% tmp2 = x(:,:,:,id+1:end);
tmp2 = x(:,:,:,:,idx2(:));
out = cat(3,tmp1,tmp2);

end

