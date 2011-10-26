
Xbase = data(:,1:2);
y = data(:,3);
m = length(y);
[X_norm, mu, sigma] = featureNormalize(Xbase);

mu
sigma

X=[ones(m,1), X_norm];

alpha = 0.01;
theta0 = zeros(3,1);
num_iters = 1000;

[theta_norm, J_history] = gradientDescentMulti(X, y, theta0, alpha, num_iters); 


t2 = theta_norm(2:3) - mu' ;
t3 = t2 ./ sigma';

theta = [theta(1); t3];

theta_norm2 = normalEqn(X, y);

theta_norm
theta_norm2
diff = theta_norm2 - theta_norm

