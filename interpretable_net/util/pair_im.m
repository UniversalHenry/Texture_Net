function out = pair_im( x, idx )
s = size(x);
tmp1 = repmat(x(:,:,:,idx),[1,1,1,s(4)-1]);
tmp2 = x(:,:,:,setdiff(1:s(4),idx));
output = cat(3,tmp1,tmp2);
end

