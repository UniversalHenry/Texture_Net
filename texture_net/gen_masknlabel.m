function [r1,c1,r2,c2,label] = gen_masknlabel(label_img, imax)
r1=randi(imax);
c1=randi(imax);
shouldRechoose=true;
while shouldRechoose
    shouldRechoose=false;
    r2=randi(imax);
    c2=randi(imax);
    if (r2==r1) && (c2==c1)
        shouldRechoose=true;
    end
end

%enlarge coordinate
s=size(label_img);
rr1 = uint16(r1/imax*s(1));
cc1 = uint16(c1/imax*s(2));

rr2 = uint16(r2/imax*s(1));
cc2 = uint16(c2/imax*s(2));

l1 = label_img(rr1,cc1,1);
l2 = label_img(rr2,cc2,1);


if l1==l2
    label=1;
else
    label=-1;
    
end

