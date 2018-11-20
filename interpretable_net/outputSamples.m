function outputSamples( x, fmap, labels, peaks, peak_coord, tau, side )
tmp = find(labels>=0);
[~,~,filter,sample] = ind2sub(size(labels), tmp(randi(length(tmp))));

[~, ind] = max(peaks, [], 4);
top_ind = reshape(ind, [size(ind,1),size(ind,2),size(ind,3),1]);
s = top_ind(:,:,filter,:);

[hh,ww] = ndgrid((1:14)-double(peak_coord(1,filter,sample)),(1:14)-double(peak_coord(2,filter,sample)));
m = single( (abs(hh)<side/227*14/2)&(abs(ww)<side/227*14/2) );
new_mask = ( single(fmap(:,:,filter,sample)>tau) + m ) > 0;
im = gather(uint8(repmat(imresize(gather(new_mask),[227,227],'bilinear'),[1,1,3]).*(x(:,:,:,sample)+128)));
im = insertText(im,[1,50],gather(labels(1,1,filter,sample)),'AnchorPoint','LeftBottom');
figure; imshow(im);

[hh,ww] = ndgrid((1:14)-double(peak_coord(1,filter,s)),(1:14)-double(peak_coord(2,filter,s)));
m = single( (abs(hh)<side/227*14/2)&(abs(ww)<side/227*14/2) );
new_mask = ( single(fmap(:,:,filter,s)>tau) + m ) > 0;
im = gather(uint8(repmat(imresize(gather(new_mask),[227,227],'bilinear'),[1,1,3]).*(x(:,:,:,s)+128)));
im = insertText(im,[1,50],gather(labels(1,1,filter,s)),'AnchorPoint','LeftBottom');
figure; imshow(im);


end

