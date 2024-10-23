
function [accuracy]=Accuracy(test_targets,predict_targets)
% syntax
%   [accuracy]=Accuracy(test_targets,predict_targets)
%
% input
%   test_targets        - L x num_test data matrix of groundtruth labels
%   predict_targets     - L x num_test data matrix of predicted labels

    [~,num_test]=size(test_targets);
    test_targets=double(test_targets==1);
    predict_targets=double(predict_targets==1);
    
    accuracy=0;
    
    for i=1:num_test
        intersection=test_targets(:,i)'*predict_targets(:,i);
        union=sum(or(test_targets(:,i),predict_targets(:,i)));        
        if union~=0
            accuracy=accuracy + intersection/union;
        end
    end
    
    accuracy=accuracy/num_test;
end