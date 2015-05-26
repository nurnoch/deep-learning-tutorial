# 向量化（Vectorization）

前面我们使用线性回归预测房价，因为数据量比较小，所以对代码的运行时间要求不高。在前面的两个练习中，我们通过循环遍历每一个样本点，求出目标函数和其梯度的值，当数据量很大的时候循环就会非常慢（因为使用MATLAB遍历序列速度是非常慢的）。为了加快MATLAB代码执行的速度，应该避免循环，使用经过优化的向量和矩阵操作，这些操作在MATLAB中的执行速度都很快。

以下通过几个例子来说明在MATLAB中如何向量化各种操作。


###许多矩阵向量积（Many matrix-vector products）
例如需要计算每一个样本点$x^{(i)}$与参数$\theta$的向量积：$\theta^Tx^{(i)}$。循环的方法是遍历所有样本，依次计算每个样本点的向量积。我们可以将所有样本点当做一个列向量构成一个矩阵X：
$$X = \left[\begin{array}{cccc}
  | & |  &  | & | \\
  x^{(1)} & x^{(2)} & \cdots & x^{(m)}\\
    | & |  &  | & |\end{array}\right]$$
这样，我们就能一次性计算所有$x^{(i)}对应的$$y^{(i)}=Wx^{(i)}$的值：
$$\left[\begin{array}{cccc}
| & |  &  | & | \\
y^{(1)} & y^{(2)} & \cdots & y^{(m)}\\
| & |  &  | & |\end{array}\right] = Y = W X$$
因此，在实现线性回归的时候，我们可以使用$\theta^Tx^{(i)}$来代替循环遍历所有的样本点计算$y^{(i)}=\theta^TX^{(i)}$。


###标准化多个向量（normalizing many vectors）
矩阵X是由多个列向量$x^{(i)}$组成的，如果想对X标准化，可以：
```matlab
X_norm = sqrt( sum(X .^ 2, 1) );  
Y = bsxfun(@rdivide, X, X_norm);
```
以上代码，首先计算矩阵X中每一列的平方和，然后再对每一个元素求平方根，形成一个1xm的矩阵 。然后利用bsxfun函数对所有列向量$x^{(i)}$标准化。
```matlab
>> X

X =

     3     7     5
     0     4     2
     1     2     3
     
>> X_norm = sqrt(sum(X .^ 2, 1))

X_norm =

    3.1623    8.3066    6.1644
     
>> Y = bsxfun(@rdivide, X, X_norm)

Y =

    0.9487    0.8427    0.8111
         0    0.4815    0.3244
    0.3162    0.2408    0.4867
```

###梯度计算中的矩阵相乘
在线性回归中，我们使用利用以下式子计算梯度：
$$\frac{\partial J(\theta; X,y)}{\partial \theta_j} = \sum_i x_j^{(i)} (\hat{y}^{(i)} - y^{(i)}).$$
对于形如以上固定某一个下标（如j），计算另一个下标的和（如i）的形式，我们可以将矩阵相乘的形式写成：
$$[AB]_{jk}=\sum_iA_{ji}B_{ik}$$
如果y和$\hat y$都是列向量，那么我们可以将梯度的计算改写为以下形式：
$$\frac{\partial J(\theta; X,y)}{\partial \theta_j} = \sum_iX_{ji}(\hat{y}_i-y_i)=[X(\hat{y}-y)]_j$$

对应的MATLAB代码：
```matlab
% X(j,i) = j'th coordinate of i'th example
% y(i) = i'th value to predicted; y is a column vector.
% theta = vector of parmameters; theata is a column vector

y_hat = theta' * X; % so that y_hat(i) = theta' * X(:,i). y_hat is a row vector
g = X * (y_hat' - y);
```

###实现线性回归和逻辑回归的向量化版本
**linear regression:**
```matlab
function [f,g] = linear_regression_vec(theta, X,y)
  %
  % Arguments:
  %   theta - A vector containing the parameter values to optimize. n*1 matrix 
  %   
  %   X - The examples stored in a matrix.  n * m matrix
  %       X(i,j) is the i'th coordinate of the j'th example.
  %   y - The target value for each example.  y(j) is the target for
  %   example j.   1 * m matrix
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
```
运行时间很快：
```
Reached Maximum Number of Iterations
Optimization took 0.089627 seconds.
RMS training error: 4.633630
RMS testing error: 4.878475
```

**logistic regression:**
```matlab
function [f,g] = logistic_regression_vec(theta, X,y)
  %
  % Arguments:
  %   theta - A column vector containing the parameter values to optimize.
  %   X - The examples stored in a matrix.  
  %       X(i,j) is the i'th coordinate of the j'th example.
  %   y - The label for each example.  y(j) is the j'th example's label.
  %
  m=size(X,2);
  
  % initialize objective value and gradient.
  f = 0;
  g = zeros(size(theta));
  

  %
  % TODO:  Compute the logistic regression objective function and gradient 
  %        using vectorized code.  (It will be just a few lines of code!)
  %        Store the objective function value in 'f', and the gradient in 'g'.
  %
  f = f - sum(y .* log(sigmoid(theta' * X)) + (1 - y) .* log(1 - sigmoid(theta' * X)));
  g = g + X * (sigmoid(theta' * X) - y)';
end
```
运行时间也比循环的版本提高了接近10倍：
```
Step Size below progTol
Optimization took 8.600933 seconds.
Training accuracy: 100.0%
Test accuracy: 100.0%
```

