function base_img = overlay_helper(base_img, img, r, c, radius)

imageSize = size(base_img);
[xx,yy] = ndgrid((1:imageSize(1))-r,(1:imageSize(2))-c);
mask = uint16((xx.^2 + yy.^2)>radius^2);
base_img(:,:,1) = base_img(:,:,1).*mask;
base_img(:,:,2) = base_img(:,:,2).*mask;
base_img(:,:,3) = base_img(:,:,3).*mask;
rlow = max([r-radius+1,1]);
rup = min([r+radius,imageSize(1)]);
clow = max([c-radius+1,1]);
cup = min([c+radius,imageSize(2)]);

imrlow = max([1,radius-r+1]);
imheight = radius*2-max([r+radius-imageSize(1), 0])-max([1,radius-r+1]);
imclow = max([1,radius-c+1]);
imwidth = radius*2-max([c+radius-imageSize(2), 0])-max([1,radius-c+1]);

img = imcrop(img,[imclow imrlow imwidth imheight]);
img = img .* uint16(base_img(rlow:rup, clow:cup, :) == 0);
base_img(rlow:rup, clow:cup, :) = base_img(rlow:rup, clow:cup, :) + img;

end