function outputImage = img_resize( inputImages,index,newSize,side )
%# Initializations:
ind1 = index{1};
ind2 = index{2};

uh = min(ceil(side/2) + ind1-1, newSize(1));
lh = max(ind1-floor(side/2), 1);
uw = min(ceil(side/2) + ind2-1, newSize(2));
lw = max(ind2-floor(side/2), 1);

% change so that height and width are 'side'-length
uh(lh==1) = min(lh(lh==1)+side-1, newSize(1));
uw(lw==1) = min(lw(lw==1)+side-1, newSize(2));
lh(uh==newSize(1)) = max(uh(uh==newSize(1))-side+1,1);
lw(uw==newSize(2)) = max(uw(uw==newSize(2))-side+1,1);

s = size(uh);
origs = size(inputImages);

rowIndex = repmat(reshape((1:side)', [side,1]),[1,s(2)]);
rowIndex = rowIndex + double(repmat(lh,[side,1])) - 1;

colIndex = repmat(reshape((1:side)', [side,1]),[1,s(2)]);
colIndex = colIndex + double(repmat(lw,[side,1])) - 1;

rind = repmat(reshape(rowIndex,[side,1,s(2)]),[1,side,1]);
cind = repmat(reshape(colIndex,[1,side,s(2)]),[side,1,1]);
index = repmat(reshape( rind + (cind-1).*origs(1), [side, side, 1, s(2)]), [1,1,origs(3),1]);
index = index + repmat(reshape(origs(1)*origs(2)*(0:origs(3)-1), [1,1,origs(3),1]), [side, side, 1, s(2)]);
index = index + repmat(reshape(origs(1)*origs(2)*origs(3)*(0:s(2)-1), [1,1,1,s(2)]), [side,side, origs(3), 1]);

% Index old image to get new image

outputImage = inputImages(index);
outputImage = imresize(outputImage, newSize);
end

