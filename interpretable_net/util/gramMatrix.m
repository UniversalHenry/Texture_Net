function gram = gramMatrix( fmap )
s=size(fmap);
F1 = reshape(fmap,[s(1)*s(2),s(3),s(4)]);
F2 = permute(conj(F1),[2,1,3]);
gram = gpuArray(zeros(s(3),s(3),s(4),'single'));
for k=1:s(4)
    gram(:,:,k) = F2(:,:,k)*F1(:,:,k);
end
end