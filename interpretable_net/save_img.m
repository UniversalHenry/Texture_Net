  function save_img( saveDir, images, feature_map,class, labels, peak_coord, tau, curr_batch ,side,imdbMean )
  if ~exist(saveDir, 'dir')
      mkdir(saveDir);
  end
    is=size(images);
    fs=size(feature_map);
    peak_coord(1,:,:) = gather(min(max(peak_coord(1,:,:).*is(1)./fs(1)-side/2,1),is(2)-side));
    peak_coord(2,:,:) = gather(min(max(peak_coord(2,:,:).*is(2)./fs(2)-side/2,1),is(2)-side));
  for fn=1:fs(3)
      dir = fullfile(saveDir, sprintf('filter_%d',fn));
      if ~exist(dir, 'dir')
        mkdir(dir);
      end
      for bn=1:fs(4)
          label = gather(labels(1,1,fn,bn));
          classscore = gather(class(:,:,:,bn));
    %           new_mask = single(feature_map(:,:,j,k)>tau);
    %           new_mask = single(imresize(gather(new_mask),[227,227],'bilinear')>0.5);
    %           new_mask(new_mask<0.5) = 0.1;
           im = gather(uint8((images(:,:,:,bn)+imdbMean)));
           dir1 = dir;
          if label >= -1
              im = insertText(im,[1,50],label,'AnchorPoint','LeftBottom');
              if label > 0.8
                dir1 = fullfile(dir,'0.8');
                im = insertShape(im, 'Rectangle',[gather([peak_coord(2,fn,bn),peak_coord(1,fn,bn)]),side,side], 'color', {'green'},'LineWidth',3);
              elseif label > 0
                dir1 = fullfile(dir,'0');
                im = insertShape(im, 'Rectangle',[gather([peak_coord(2,fn,bn),peak_coord(1,fn,bn)]),side,side], 'color', {'yellow'},'LineWidth',3);
              else
                dir1 = fullfile(dir,'-1');
                im = insertShape(im, 'Rectangle',[gather([peak_coord(2,fn,bn),peak_coord(1,fn,bn)]),side,side], 'color', {'red'},'LineWidth',3);
              end
          end
             if ~exist(dir1, 'dir')
                    mkdir(dir1);
             end
              im = insertText(im,[1,150],classscore,'AnchorPoint','LeftBottom');
              imwrite(im, fullfile(dir1, sprintf('batch_%d_%d_p1.jpg',curr_batch,bn)));
      end
  end
end
  

