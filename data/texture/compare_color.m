function di = compare_color( image_, color_measure, img_idx )
R = mean(mean(image_(:,:,1)));
G = mean(mean(image_(:,:,2)));
B = mean(mean(image_(:,:,3)));
color_measure(:,img_idx) = [R;G;B];
c = repmat([R;G;B],1,4);
d = sqrt(sum((color_measure-c).^2));%color vector distance
di = sum(d(1:img_idx-1)<90); %80 is the difference threshold

end

