dirin = './data/';
dirout = './sample_check/';
images = dir([dirin,'*.mat']);
struct_size = size(images);
num_images = struct_size(1);

addpath(dirin);
for i=1:num_images
    im=load(fullfile(dirin, sprintf('img_%d.mat',i)));
    im=struct2cell(im);
    im=im{1};
    imwrite(im(:,:,1:3),fullfile(dirout, sprintf('img_%d(1).jpg',i)));
    imwrite(im(:,:,4:6),fullfile(dirout, sprintf('img_%d(2).jpg',i)));
    fprintf('img_%d.jpg\n',i);
end