function [f,g] = softmax_regression(theta, X,y)
  %
  % Arguments:
  %   theta - A vector containing the parameter values to optimize.
  %       In minFunc, theta is reshaped to a long vector.  So we need to
  %       resize it to an n-by-(num_classes-1) matrix.
  %       Recall that we assume theta(:,num_classes) = 0.
  %
  %   X - The examples stored in a matrix.  
  %       X(i,j) is the i'th coordinate of the j'th example.
  %   y - The label for each example.  y(j) is the j'th example's label. 
  %   1 x m 
  %
  m=size(X,2);
  n=size(X,1);

  % theta is a vector;  need to reshape to n x num_classes.
  theta=reshape(theta, n, []);
  num_classes=size(theta,2)+1;
  theta = [theta, zeros(size(theta,1),1)]; % n x num_classes(785 * 10)
  % initialize objective value and gradient.
  f = 0;
  g = zeros(size(theta));

  %
  % TODO:  Compute the softmax objective function and gradient using vectorized code.
  %        Store the objective function value in 'f', and the gradient in 'g'.
  %        Before returning g, make sure you form it back into a vector with g=g(:);
  %
%%% YOUR CODE HERE %%%
  
  h = exp(theta' * X);  % num_classes x m (10 * 60000)
  I = sub2ind(size(h), y, 1:size(h,2));
  f = -sum(log(h(I) ./ sum(h,1)));
  
  p = bsxfun(@rdivide, h, sum(h,1)); % k x m(10 * 60000)
  val_k = full(sparse(y, 1:m, 1)); % k x m
  g = g - X * (val_k - p)'; % n x n
  g(:,end) = [];
  g=g(:); % make gradient a vector for minFunc
end

