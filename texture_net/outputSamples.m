function outputSamples( im, labels, predictions, idx )

figure;imshow(uint8(im(:,:,1:3,idx)+128));
figure;imshow(uint8(im(:,:,4:6,idx)+128));

fprintf('label: %d\n',labels(:,:,:,idx));
fprintf('prediction: %d\n',predictions(:,:,:,idx));

end

