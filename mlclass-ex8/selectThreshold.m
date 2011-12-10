function [bestEpsilon bestF1] = selectThreshold(yval, pval)
%SELECTTHRESHOLD Find the best threshold (epsilon) to use for selecting
%outliers
%   [bestEpsilon bestF1] = SELECTTHRESHOLD(yval, pval) finds the best
%   threshold to use for selecting outliers based on the results from a
%   validation set (pval) and the ground truth (yval).
%

bestEpsilon = 0;
bestF1 = 0;
F1 = 0;

n = size(yval)

stepsize = (max(pval) - min(pval)) / 1000;
for epsilon = min(pval):stepsize:max(pval)
    
    % ====================== YOUR CODE HERE ======================
    % Instructions: Compute the F1 score of choosing epsilon as the
    %               threshold and place the value in F1. The code at the
    %               end of the loop will compare the F1 score for this
    %               choice of epsilon and set it to be the best epsilon if
    %               it is better than the current choice of epsilon.
    %               
    % Note: You can use predictions = (pval < epsilon) to get a binary vector
    %       of 0's and 1's of the outlier predictions

    pred = pval < epsilon;
    
    nyval = 1 - yval;
    npred = 1 - pred;
    
    tp = sum( pred &  yval);
    tn = sum(npred & nyval);
    fp = sum( pred & nyval);
    fn = sum(npred &  yval);
    %total=tp+tn+fp+fn
    
    precision = 0;
    recall = 0;
    F1 = 0;
    
    if tp+fp > 0
        precision = tp/(tp+fp);
    end    
    if tp+fn > 0
        recall = tp/(tp+fn);
    end
    if precision + recall > 0
        F1 = 2 * precision * recall / (precision + recall) ;   
    end
    
    % =============================================================

    if F1 > bestF1
       bestF1 = F1;
       bestEpsilon = epsilon;
    end
end

end
