function [ Result ] = evalt(Fpred, Ygnd, thr, flag)
%%
% Fpred: L*N predicted values
% Ypred: L*N predicted labels
% Ygnd: L*N groundtruth labels
% thr: threshold value
% flag: default value is true
%%
if flag
    % default
    Ypred = sign(Fpred);
else
    Ypred = sign(Fpred-thr);
end
Result.Accuracy =Accuracy(Ygnd, Ypred);
Result.Fmeasure = Fmeasure(Ygnd, Ypred);
Result.MacroF1 = Macro_F1(Ygnd, Ypred);
Result.MicroF1 = Micro_F1(Ygnd, Ypred);
Result.ExactMatch = Exact_match(Ypred,Ygnd);
