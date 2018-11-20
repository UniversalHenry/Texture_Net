function iou = calculateIOU( label_img1, label_img2, label_set )

label_img1 = label_img1(:,:,1);
label_img2 = label_img2(:,:,1);

I = 0;
U = 0;
for k=1:length(label_set)
    l = label_set(k);
    tmp1 = single(label_img1==l);
    tmp2 = single(label_img2==l);
    area1 = sum(sum(tmp1));
    area2 = sum(sum(tmp2));
    common = single((tmp1==tmp2)&(tmp1==1));
    I_ = sum(sum(common));
    U_ = area1+area2-I;
    I = I + I_;
    U = U + U_;
    
end

iou = I/U;

end

