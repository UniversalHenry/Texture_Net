function scores = calculate_score(img, maxlabel)
s = size(img);
if s(3)>1
   img = img(:,:,1);
end

scores=zeros(1,maxlabel);
num_zero_pix = sum(sum(img==0), 'double');
total_pix = s(1)*s(2)-num_zero_pix;
for k=1:maxlabel
    scr = sum(sum(img==k), 'double');
    scores(k) = scr/total_pix;
end
