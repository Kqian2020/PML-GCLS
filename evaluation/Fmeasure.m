
function [F1]=Fmeasure(test_targets,predict_targets)
% syntax
%   [ExampleBasedAccuracy,ExampleBasedPrecision,ExampleBasedRecall,ExampleBasedFmeasure]=ExampleBasedMeasure(test_targets,predict_targets)
%
% input
%   test_targets        - L x num_test data matrix of groundtruth labels
%   predict_targets     - L x num_test data matrix of predicted labels

    [~,num_test]=size(test_targets);
    test_targets=double(test_targets==1);
    predict_targets=double(predict_targets==1);
    
    F1=0;
    
    for i=1:num_test
        intersection=test_targets(:,i)'*predict_targets(:,i);

        if sum(predict_targets(:,i))~=0
            precision_i = intersection/sum(predict_targets(:,i));
        else
            precision_i=0;
        end
        if sum(test_targets(:,i))~=0
            recall_i = intersection/sum(test_targets(:,i));
        else
            recall_i=0;
        end
        if recall_i~=0 || precision_i~=0
            F1=F1 + 2*recall_i*precision_i/(recall_i+precision_i);
        end
    end
   
    F1=F1/num_test;

end