function gramlabel = get_gram_score(x_,net)
res1 = vl_simplenn(net, x_(1:224,1:224,1:3,:)) ;
fmap1 = res1(12).x;
res2 = vl_simplenn(net, x_(1:224,1:224,4:6,:)) ;
fmap2 = res2(12).x;
F1=gramMatrix(fmap1);
F2=gramMatrix(fmap2);
F=(F1-F2).^2./(2*13*13*256).^2;
gramlabel=sum(sum(F,1),2);
clear F F1 F2 fmap1 fmap2 res1 res2;
gramlabel=vl_nnsigmoid(6-0.002*gramlabel) * 2 - 1;
end