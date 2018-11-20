function [base_img, label_img] = overlay(base_img, img, label_img, sublabel_img, ...
    radius, num_overlay, img_idx)
imageSize=size(base_img);
imgsize = size(img);
if (imgsize(1)~= 2*radius) || (imgsize(2)~= 2*radius)
    img = imresize(img,[2*radius,2*radius]);
end
assert(img_idx<5)
for i=1:num_overlay
    [r,c] = random_rc(imageSize(1), imageSize(2), img_idx, i);
    
    base_img = overlay_helper(base_img, img, r, c, radius);
    label_img = overlay_helper(label_img, sublabel_img, r, c, radius);

end


end


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

function [r,c] = random_rc(height, width, img_idx, overlay_idx)
% making strong assumption about number of areas to separate
idx = mod(int8(img_idx+overlay_idx-2),4)+1;
min_r = 0 + idivide(int32(idx-1), 2) * height/2;
max_r = min_r + height/2;
min_c = 0 + mod(int32(idx-1), 2) * width/2;
max_c = min_c + width/2;
r = randi([min_r, max_r]);
c = randi([min_c, max_c]);
end
