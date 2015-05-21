function [f,g] = linear_regression_vec(theta, X,y)
  %
  % Arguments:
  %   theta - A vector containing the parameter values to optimize. n*1 matrix 
  %   
  %   X - The examples stored in a matrix.  n * m matrix
  %       X(i,j) is the i'th coordinate of the j'th example.
  %   y - The target value for each example.  y(j) is the target for
  %   example j. 
  %
  m=size(X,2);
  
  % initialize objective value and gradient.
  f = 0;
  g = zeros(size(theta));

  %
  % TODO:  Compute the linear regression objective function and gradient 
  %        using vectorized code.  (It will be just a few lines of code!)
  %        Store the objective function value in 'f', and the gradient in 'g'.
  %
  f = f + 0.5 * sum((theta' * X - y) .^ 2);  
  g = g + X * (theta' * X - y)';
end