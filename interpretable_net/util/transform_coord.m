function [r1, c1]=transform_coord(h1,w1,imax,s)
r1 = uint16(single(h1)./imax(1).*s(1)-s(1)/imax(1)/2);
c1 = uint16(single(w1)./imax(2).*s(2)-s(2)/imax(2)/2);
r1 = min(max(r1,1),s(1));
c1 = min(max(c1,1),s(2));
end
