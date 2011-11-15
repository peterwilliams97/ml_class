function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

%X
%y
%Theta1
%Theta2

yb = zeros(m,num_labels);
for i=1:m
 yb(i,y(i)) = 1;
end 
X1 = [ones(m,1), X];

z1 = X1 * Theta1';
a1 = sigmoid(z1);
a11 = [ones(m,1), a1];
z2 = a11 * Theta2';
a2 = sigmoid(z2);

%[_,p] = max(a2, [], 2);
%size(a1)
%size(a2)

% J = 1/m(-y*log(h*x) - (1-y)log(1-h*x)) + (lambda/2m)(theta1^2+ theta2^2)
for i = 1:m
 for k = 1:num_labels
  p = -yb(i,k)*log(a2(i,k)) - (1-yb(i,k))*log(1-a2(i,k));
  J = J + p;
 end 
end
J = J/m;

% Compute the regularization penalty
reg_penalty = 0;

for j = 1:size(Theta1)(1)
 for k = 2:size(Theta1)(2)
    reg_penalty = reg_penalty + Theta1(j,k)*Theta1(j,k);
 end 
end

for j = 1:size(Theta2)(1)
 for k = 2:size(Theta2)(2)
    reg_penalty = reg_penalty + Theta2(j,k)*Theta2(j,k);
 end 
end

reg_penalty = reg_penalty*lambda/(2*m);

J = J + reg_penalty;

%J = (1/m)*(-yb.*log(a2) - (1-yb).*log(1-a2));


% -------------------------------------------------------------
% Compute gradients using back propagation

% Sample by sample
for t = 1:m
    % Forward to calculate error for sample t
    a_1 = X(t,:)';
    a_1 = [1; a_1];
           
    z_2 = Theta1 * a_1;
    
    a_2 = sigmoid(z_2);
    a_2 = [1; a_2];
    
    z_3 = Theta2 * a_2;
    
    a_3 = sigmoid(z_3);
    
    yt = yb(t,:)';
    delta_3 = a_3 - yt; 
    
    %size(delta_3)
    
    % Propagate error backwards
    delta_2 = (Theta2' * delta_3) .* sigmoidGradient([1; z_2]);
    delta_2 = delta_2(2:end); 
    %size(delta_2)
 
  %  delta_1 = (delta_2 * Theta1);
  %  delta_1 = delta_1; 
  %  delta_1 = delta_1 .* sigmoidGradient(a_1);
    %size(delta_1)
 
%    delta_1 = (delta_2 * Theta1) .* sigmoidGradient(xt);
%    delta_1 = delta_1(2:end); 
%    size(delta_1)
    
    %size(a2)
    %size(delta_3)
    %size(Theta2_grad)

    dt2 = delta_3 * a_2';
    %size(dt2)
    dt1 = delta_2 * a_1';
    %size(dt1)
    
 
    Theta2_grad = Theta2_grad + dt2;
    Theta1_grad = Theta1_grad + dt1;
   
end

Theta1_grad = (1/m) * Theta1_grad;
Theta1_grad(:,2:end) = Theta1_grad(:,2:end) + (lambda/m) * Theta1(:,2:end);
Theta2_grad = (1/m) * Theta2_grad ;
Theta2_grad(:,2:end) = Theta2_grad(:,2:end) + (lambda/m) * Theta2(:,2:end);

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
