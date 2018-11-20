function output = get_peak_area(x,fmap, peaks, peak_coord, side,outputSides )
is=size(x);
fs=size(fmap);
filter_num = size(peaks,3);
ind1 = peak_coord(1,:,:);
ind2 = peak_coord(2,:,:);
ind1 = min(max(ind1.*is(1)./fs(1)-side/2,1),is(2)-side);
ind2 = min(max(ind2.*is(2)./fs(2)-side/2,1),is(2)-side);
output = gpuArray(zeros(side,side,outputSides(3),filter_num));
for i = 1:filter_num
    output(:,:,1:3,i) = x(ind1(:,i,1):ind1(:,i,1)+side-1,ind2(:,i,1):ind2(:,i,1)+side-1,:,1);
    output(:,:,4:6,i) = x(ind1(:,i,2):ind1(:,i,2)+side-1,ind2(:,i,2):ind2(:,i,2)+side-1,:,2);
end
output = imresize(output,[outputSides(1),outputSides(2)]);
output = cat(4,output,gpuArray(zeros(outputSides(1),outputSides(2),outputSides(3),filter_num)));
output(:,:,1:3,filter_num+1:end)=output(:,:,4:6,1:filter_num);
output(:,:,4:6,filter_num+1:end)=output(:,:,1:3,1:filter_num);
end