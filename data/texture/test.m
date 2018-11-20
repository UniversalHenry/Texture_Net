
% images = dir(['./data2/*.mat']);
% for i=22:22+length(images)
%     img = load(['./data2/',images(i-21).name]);
%     img1 = img.new_img(:,:,1:3);
%     img2 = img.new_img(:,:,4:6);
%     imwrite(img1,sprintf('t%d1.jpg',i));
%     imwrite(img2,sprintf('t%d2.jpg',i));
% end    

% images = dir(['./t*.jpg']);
for i=11:21
    t1=imread(sprintf('t%d1.jpg',i));
    t2=imread(sprintf('t%d2.jpg',i));
    t1 = imresize(t1,[227,227]);
    t2 = imresize(t2,[227,227]);
    new_img1 = cat(3,t1,t2);
    save(sprintf('./data3/img_%d.mat',i),'new_img1');
end
