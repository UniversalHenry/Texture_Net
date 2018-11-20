texture_dir = '/media/vclagpu/Data12/zhou/My_Items/NetDissect/dataset/broden1_224/images/dtd/';

images = dir([texture_dir,'*.jpg']);
radius_list = [160,180,200];
shouldOverlay = true;
struct_size = size(images);
num_images = struct_size(1);

my_texture_dir = 'data/';
my_images = dir([my_texture_dir,'*.jpg']);
start = 1;

for i=start:start+10
    
    shouldReprocess=true;
    while shouldReprocess
        shouldReprocess=false;
        texture_num = 4;
        common_num1 = randi(texture_num/2);
        common_num2 = randi(texture_num/2);
        
        bg1 = randi(num_images);
        bg2 = randi(num_images);
        %base texture for 1 / texture of 2
        new_img1 = im2uint16(imread(fullfile(texture_dir, images(bg1).name)));
        new_img1 = map_image(uint16(500*sqrt(3)), new_img1);
        label_img1 = uint16(repmat([bg1],uint16(500*sqrt(3)),uint16(500*sqrt(3)),3));
        %base texture for 2 / texture of 1
        new_img2 = im2uint16(imread(fullfile(texture_dir, images(bg2).name)));
        new_img2 = map_image(uint16(500*sqrt(3)), new_img2);
        label_img2 = uint16(repmat([bg2],uint16(500*sqrt(3)),uint16(500*sqrt(3)),3));
        
        new_imgs = {new_img1, new_img2};
        label_imgs = {label_img1, label_img2};
        
        
        common_texture = randi(num_images, 1, common_num1 + common_num2);
        
        for j = 1:2
            new_img1 = new_imgs{j};
            label_img1 = label_imgs{j};
            for l = 1:length(common_texture) % textures
                id = common_texture(l);
                image1 = im2uint16(imread(fullfile(texture_dir, images(id).name)));
                sublabel_img1 = uint16(repmat(id, size(image1,1), size(image1,2), 3));
                im1 = crop_img(image1);
                sublabel_im1 = crop_img(sublabel_img1);
                r1 = radius_list(randi(3));
                im1 = imresize(im1, [2*r1, 2*r1]);
                sublabel_im1 = imresize(sublabel_im1, [2*r1, 2*r1]);
                [new_img1, label_img1] = overlay(new_img1, im1, label_img1, sublabel_im1, r1, 4, 1);
                new_imgs{j} = new_img1;
            end
            label_imgs{j} = label_img1;
        end
        
        label_set = {cat(2,[bg1],common_texture), cat(2,[bg2],common_texture)}; % 1 has all the labels in the first img and same for 2
        
        %randomly select images from the file
        randidxgroup = {randperm(num_images, texture_num-common_num1), randperm(num_images, texture_num-common_num2)};
        
        for j = 1:2
            randidx = randidxgroup{j};
            img_idx = 1;
            color_measure = zeros(3,4);

            for img = images(randidx)'

                fprintf('Processing image: %s\n', img.name);
                tmp_img = imread(fullfile(texture_dir, img.name));
                image = im2uint16(tmp_img);
                image_ = double(tmp_img); %conversion to double 
                %compare color of the images
                di = compare_color(image_, color_measure, img_idx);
                if di~=0
                    shouldReprocess=true;
                    break;
                end
                im = crop_img(image);
                fprintf('Cropping complete.\n');
                
                tmp_list = cat(2, [80, 100,140], radius_list); % more option for other textures
                ri = randi(length(tmp_list));
                r = tmp_list(ri);
                fprintf('Radius selected: %d\n', r);

                im = imresize(im, [2*r, 2*r]);
                l = uint16(repmat([randidx(img_idx)],2*r,2*r,3));
                %overlay
                %start overlay
                fprintf('Number to overlay on top of the texture image: %d\n', 3);
                [new_imgs{j}, label_imgs{j}] = overlay(new_imgs{j}, im, label_imgs{j}, l, r, 3, img_idx);
                %update label set
                label_set{j} = union(label_set{j}, [randidx(img_idx)]);
                fprintf('Overlay completed.\n\n');
                img_idx = img_idx + 1;

            end
            
        end
    end
    % process and calculate IOU
    if randi(30)==1 % 1/30 chance of having the same texture
        new_imgs{2}=new_imgs{1};
        label_imgs{2}=label_imgs{1};
        label = 1;
    else
        label_set = intersect(label_set{1},label_set{2});
        label = calculateIOU(label_imgs{1}, label_imgs{2}, label_set);
    end
    new_img1 = im2uint8(imresize(new_imgs{1}, [227,227]));
    new_img2 = im2uint8(imresize(new_imgs{2}, [227,227]));
    new_img = cat(3, new_img1, new_img2);
    
    store_path = sprintf('data/img_%d.mat',i);
    label_path = sprintf('label/img_%d.mat',i);
    %saving
    fprintf('Saving new images.\n\n');
    save(fullfile(pwd,store_path), 'new_img');
%     if ~exist('data_img', 'dir')
%         mkdir('data_img');
%     end
%     imgdir = sprintf('data_img/img_%d',i);
%     if ~exist(imgdir, 'dir')
%         mkdir(imgdir);
%     end
%     imwrite(im2uint8(new_img1), fullfile(imgdir,'1.jpg'));
%     imwrite(im2uint8(new_img2), fullfile(imgdir,'2.jpg'));
    save(fullfile(pwd,label_path), 'label');

end


