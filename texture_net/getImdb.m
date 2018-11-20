% --------------------------------------------------------------------
function imdb = getImdb(dataDir, labelDir, IsTrain)
% --------------------------------------------------------------------
% Initialize the imdb structure (image database).
% Note the fields are arbitrary: only your getBatch needs to understand it.
% The field imdb.set is used to distinguish between the training and
% validation sets, and is only used in the above call to cnn_train.

% The sets, and number of samples per label in each set
files = dir(fullfile(dataDir, '*.mat'));

sets = {'train', 'val', 'test'} ;
train_rate = 0.7;
val_rate = 0.2;

% Preallocate memory
totalSamples = length(files);
num_img = totalSamples;
images = zeros(227, 227, 6, totalSamples, 'single') ;
labels = zeros(1, 1, 1, totalSamples, 'single') ;
set = ones(1, totalSamples) ;

% Read all samples
sample = 1 ;

counter = 1;
% iterate through positive images
set_label = 1; % default to train


for file = files'
    im = load(fullfile(dataDir, file.name));
    im = im.new_img1;
    s = size(im);
    assert(s(1)==227);
    assert(s(2)==227);
    assert(s(3)==6);
    name_arr = strsplit(file.name, '.');
    label = load(fullfile(labelDir, [name_arr{1}, '.mat']));
    img1 = im2uint8(im(:,:,1:3));
    img2 = im2uint8(im(:,:,4:6));
    images(:,:,:,sample) = single(cat(3,img1,img2));
    labels(:,:,:,sample) = label.label;
    set(1,sample) = set_label; % 1 is train, 2 is val
    
    fprintf('Processed image: %s\n', file.name);
    fprintf('Sample number processed: %d\n', sample);
    
    sample = sample + 1;
    counter = counter + 1;
    
    if counter > (train_rate + val_rate)*num_img
        set_label = 3;
    elseif counter > train_rate*num_img
        set_label = 2;
    end
end

%shuffle set
s = size(labels, 4);
i = randperm(s);
set = set(:,i);

% Remove mean over whole dataset
dataMean = mean(images, 4);
images = bsxfun(@minus, images, dataMean) ;

% Store results in the imdb struct
imdb.images.data = images ;
imdb.images.labels = labels ;
imdb.images.set = set ;

totalSamples = size(labels,4);

if(IsTrain)
    rng(0);
    list=randperm(totalSamples);
    imdb.images.data=imdb.images.data(:,:,:,list);
    imdb.images.labels=imdb.images.labels(:,:,:,list);
    imdb.images.set=imdb.images.set(:,list);
    [~,imdb.images.order]=sort(list);
end
imdb.meta.sets = sets;
imdb.meta.classes = {'texture'};
imdb.meta.dataMean = dataMean;