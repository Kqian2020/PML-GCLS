clear;clc;
addpath(genpath('.'));
%load data
load('Birds.mat');

%% parameter
parameter.lambda1          = 10^2;
parameter.lambda2          = 10^-2;
parameter.lambda3          = 10^-3;
parameter.lambda4          = 10^-4;
parameter.lambda5          = 10^-6;
parameter.num_K            = 10;
parameter.minLoss          = 10^-4;
parameter.maxIter          = 50;
parameter.rho              = 1.1;
parameter.mu               = 10^-6;
parameter.maxMu            = 10^6;
parameter.epsilon          = 10^-6;
parameter.paraDc           = 1.2;
parameter.average_num      = 1;

%% perpare data
data    = [train_data;test_data];
target  = double([train_target,test_target]);
[DN,~] = size(data);
[~,TN] = size(target);
data = [data, ones(DN,1)];

%% cross validation
assert(DN==TN, 'Dimensional inconsistency')
runTimes = 1;
cross_num = 5;
All_results = zeros(22, cross_num*runTimes);
for r = 1:runTimes
    A = (1:DN)';
    indices = crossvalind('Kfold', A(1:DN,1), cross_num);
    for k = 1:cross_num 
        test = (indices == k);
        test_ID = find(test==1);
        train_ID = find(test==0);
        
        TE_data = data(test_ID,:);
        TR_data = data(train_ID,:);
        TE_target = target(:,test_ID);
        TR_target = target(:,train_ID);
        
        % partial label matrix
        partial_label = generatePartial(TR_target, parameter.average_num);
        
        % train
        modelTrain  = train(TR_data, partial_label', parameter);
        
        %prediction and evaluation
        zz = mean(TE_target);
        TE_target(:,zz==-1) = [];
        TE_data(zz==-1,:) = [];
        [Output,results] = Predict(modelTrain, TE_data, TE_target);
      
        All_results(1,(r-1)*cross_num+k) = results.Accuracy;
        All_results(2,(r-1)*cross_num+k) = results.ExactMatch;
        All_results(3,(r-1)*cross_num+k) = results.Fmeasure;
        All_results(4,(r-1)*cross_num+k) = results.MacroF1;
        All_results(5,(r-1)*cross_num+k) = results.MicroF1;
        
    end
end
average_std = [mean(All_results,2) std(All_results,1,2)];
PrintResults(average_std);