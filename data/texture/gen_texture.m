texture_dir = '../../NetDissect/dataset/broden1_224/images/dtd/';

images = dir([texture_dir,'*.jpg']);
radius_list = [160,180,200];
shouldOverlay = true;
struct_size = size(images);
num_images = struct_size(1);
outputSize = [64,64,3];

start = 1;

for i=start:start+num_images*3
    
    if mod(i,2) == 1
        % then negative sample
        bg = randperm(num_images, 2);
        bg1 = bg(1);
        bg2 = bg(2);
        
        %base texture for 1 / texture of 2
        img1 = im2uint16(imread(fullfile(texture_dir, images(bg1).name)));
        %base texture for 2 / texture of 1
        img2 = im2uint16(imread(fullfile(texture_dir, images(bg2).name)));
        imgs = {img1,img2};
        label = -1;

        
    else
        % positive sample
        bg1 = randi(num_images);
        
        %base texture for 1 / texture of 2
        img1 = im2uint16(imread(fullfile(texture_dir, images(bg1).name)));
        img1_large = repmat(img1,[3,3,1]);
        % random croping size
        s1 = size(img1);
        assert(s1(1)==s1(2));
        sx = randi([ceil(s1(2)/2), s1(2)]);
        sy = sx;
        xmin1 = randi(size(img1,2)-sx+1);
        ymin1 = randi(size(img1,1)-sy+1);
        img1 = imcrop(img1, [xmin1, ymin1, sx, sy]);
        img1 = imresize(img1, [outputSize(1),outputSize(2)]);
        %base texture for 2 / texture of 1
        % random rotation
        img2 = imrotate(img1_large, rand()*360, 'bilinear', 'crop'); 
        s2 = size(img2);
        xmin2 = floor(s2(2)/2-sx/2);
        ymin2 = floor(s2(1)/2-sy/2);
        img2 = imcrop(img2,[xmin2, ymin2, sx, sy]);
        img2 = imresize(img2, [outputSize(1),outputSize(2)]);
        imgs = {img1, img2};
        
        label = 1;
        
    end
    
    img1 = im2uint8(imresize(imgs{1}, [outputSize(1),outputSize(2)]));
    img2 = im2uint8(imresize(imgs{2}, [outputSize(1),outputSize(2)]));
    new_img1 = cat(3, img1, img2);
    
    store_path = sprintf('data/img_%d.mat',i);
    label_path = sprintf('label/img_%d.mat',i);
    %saving
    fprintf('Saving new images %s.\n\n',store_path);
    save(fullfile(pwd,store_path), 'new_img1');
    save(fullfile(pwd,label_path), 'label');

end



