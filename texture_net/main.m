isMultiClassClassification=false;
categoryName ='texture';
model = 'alexnet';


addpath(genpath('../matconvnet-1.0-beta25/'));
%vl_compilenn('enableGpu', true);
% vl_setupnn;

%setup data
setup_dataconf();

%learn the model
learn_alexnet(model,categoryName);

showResults(categoryName);
