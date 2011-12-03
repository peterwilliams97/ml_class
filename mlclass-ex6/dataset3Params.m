function [C, sigma] = dataset3Params(X, y, Xval, yval)
%EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

C = 0.27
sigma = 0.09
% best_error = 6

best_error = 10^10;
fac = sqrt(3);

if false
    C_x = 0.01;
    
    while C_x <= 10.0 * fac
        
        sigma_x = 0.01;
        while sigma_x <= 10.0 * fac
        
            model = svmTrain(X, y, C_x, @(x1, x2) gaussianKernel(x1, x2, sigma_x));
            predictions = svmPredict(model, Xval);
            error = sum((predictions-yval).^2);
            if error < best_error
                C = C_x
                sigma = sigma_x
                best_error = error
                
            end
            
            sigma_x = sigma_x * fac;    
        end
        
        C_x = C_x * fac;
    end    

end

C
sigma
best_error






% =========================================================================

end
