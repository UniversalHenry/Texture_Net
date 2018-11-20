function im = crop_img(img)
imageSize = size(img);
r = (imageSize(1)<imageSize(2))*(imageSize(1)/2) + (imageSize(1)>=imageSize(2))*(imageSize(2)/2);
ci = [imageSize(1)/2, imageSize(2)/2, r];     % center and radius of circle ([c_row, c_col, r])
[xx,yy] = ndgrid((1:imageSize(1))-ci(1),(1:imageSize(2))-ci(2));
mask = uint16((xx.^2 + yy.^2)<ci(3)^2);
im = uint16(zeros(size(img)));
im(:,:,1) = img(:,:,1).*mask;
im(:,:,2) = img(:,:,2).*mask;
im(:,:,3) = img(:,:,3).*mask;