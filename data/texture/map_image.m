function new_img = map_image(output_dim, img)
s=size(img);
s1len=ceil(output_dim/s(1));
s2len=ceil(output_dim/s(2));
new_img = zeros(s1len*s(1),s2len*s(2),3,'uint16');
for i=1:s1len
    for j=1:s2len
        new_img((i-1)*s(1)+1:i*s(1),(j-1)*s(2)+1:j*s(2),:)=img;
    end
end
new_img = new_img(1:output_dim,1:output_dim,:);

end

