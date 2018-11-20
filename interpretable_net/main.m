%learn the model
warning('off','all');
for k = [14]
    save myfile k;
    clear all;
    load myfile;
    delete myfile;
%      try
        isMultiClassClassification=false;
        dataset='ilsvrcanimalpart'; 
        Name_batch={'n01443537','n01503061','n01639765','n01662784','n01674464','n01882714','n01982650','n02084071','n02118333','n02121808','n02129165','n02129604','n02131653','n02324045','n02342885','n02355227','n02374451','n02391049','n02395003','n02398521','n02402425','n02411705','n02419796','n02437136','n02444819','n02454379','n02484322','n02503517','n02509815','n02510455'};
        model = 'vgg-16';
        setup_matconvnet(); % setup matconvnet
        addpath(genpath('../matconvnet-1.0-beta25/'));
        addpath(genpath('./util/'));
        vl_setupnn;
        setup_ilsvrcanimalpart();
        categoryName =Name_batch{k};
        learn_vgg(model,categoryName);
        showResults(categoryName);
%      catch
%      end
end