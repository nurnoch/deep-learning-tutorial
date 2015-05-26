# softmax 回归模型（Softmax Regression）

###介绍
Softmax regression（即多元逻辑回归）是逻辑回归（logistic regression）的一种泛化形式，用于处理多分类的问题。在逻辑回归中我们假设只有两种标签{0, 1}，而在softmax regression中我们可以处理$y^{(i)} \in \{1,\ldots,K\}$，其中K是类别的数目。

在逻辑回归中，我们有：

- m个训练样本$\{ (x^{(1)}, y^{(1)}), \ldots, (x^{(m)}, y^{(m)}) \}$
- 分类标签$y^{(i)} \in \{0,1\}$
- 假设函数（hypothesis）为：$\begin{align}
h_\theta(x) = \frac{1}{1+\exp(-\theta^\top x)}
\end{align}$
- 通过训练参数$\theta$使得代价函数$\begin{align}
J(\theta) = -\left[ \sum_{i=1}^m y^{(i)} \log h_\theta(x^{(i)}) + (1-y^{(i)}) \log (1-h_\theta(x^{(i)})) \right]
\end{align}$最小

在softmax regression中，我们目的是实现多类分类问题，因此y可以取K个不同的值：

- m个训练样本$\{ (x^{(1)}, y^{(1)}), \ldots, (x^{(m)}, y^{(m)}) \}$
- 分类标签$y^{(i)} \in \{1,\ldots,K\}$（通常下标从1开始）

假设函数（hypothesis）为：

$$\begin{align}
h_\theta(x) =
\begin{bmatrix}
P(y = 1 | x; \theta) \\
P(y = 2 | x; \theta) \\
\vdots \\
P(y = K | x; \theta)
\end{bmatrix}
=
\frac{1}{ \sum_{j=1}^{K}{\exp(\theta^{(j)\top} x) }}
\begin{bmatrix}
\exp(\theta^{(1)\top} x ) \\
\exp(\theta^{(2)\top} x ) \\
\vdots \\
\exp(\theta^{(K)\top} x ) \\
\end{bmatrix}
\end{align}$$
给定一个输入x，我们需要假设函数估计出属于每一个分类的概率$P(y=k|x)$。因此，假设函数的输出是一个K维的向量给出了属于每一个分类的概率，并且K维向量各个元素之和为1.

通常，将参数$\theta$表示为一个n x K的矩阵：
$$\theta = \left[\begin{array}{cccc}| & | & | & | \\
\theta^{(1)} & \theta^{(2)} & \cdots & \theta^{(K)} \\
| & | & | & |
\end{array}\right].$$


###代价函数
对于softmax regression其代价函数可以表示为：
$$\begin{align}
J(\theta) = - \left[ \sum_{i=1}^{m} \sum_{k=1}^{K}  1\left\{y^{(i)} = k\right\} \log \frac{\exp(\theta^{(k)\top} x^{(i)})}{\sum_{j=1}^K \exp(\theta^{(j)\top} x^{(i)})}\right]
\end{align}$$
其中，1{.}是指示函数（indicator function），当花括号中的内容为真时，表达式的值为1.

逻辑回归的代价函数也可以写为：
$$\begin{align}
J(\theta) &= - \left[ \sum_{i=1}^m   (1-y^{(i)}) \log (1-h_\theta(x^{(i)})) + y^{(i)} \log h_\theta(x^{(i)}) \right] \\
&= - \left[ \sum_{i=1}^{m} \sum_{k=0}^{1} 1\left\{y^{(i)} = k\right\} \log P(y^{(i)} = k | x^{(i)} ; \theta) \right]
\end{align}$$
softmax regression的不同之处在于是对K个不同的k值求和。

$J(\theta)$的最小值不容易直接计算出来，因此我们使用迭代的优化算法（如梯度下降法）。先计算出损失函数的梯度。对于softmax regression，属于每一个分类的概率可以表示为：
$$P(y^{(i)} = k | x^{(i)} ; \theta) = \frac{\exp(\theta^{(k)\top} x^{(i)})}{\sum_{j=1}^K \exp(\theta^{(j)\top} x^{(i)}) }$$
因此，求导可以得到：
$$\begin{align}
\nabla_{\theta^{(k)}} J(\theta) = - \sum_{i=1}^{m}{ \left[ x^{(i)} \left( 1\{ y^{(i)} = k\}  - P(y^{(i)} = k | x^{(i)}; \theta) \right) \right]  }
\end{align}$$
其中$\nabla_{\theta^{(k)}}$是一个向量，它的第j个元素$\frac{\partial J(\theta)}{\partial \theta_{jk}}$是$J(\theta)$对$\theta_k$的第j个元素的偏导数。

计算出梯度之后可以使用梯度下降等优化算法来最小化损失函数。

###softmax回归模型的参数化性质
softmax回归的一个不寻常的特点是，它具有冗余的参数。例如，我们可以将每一个参数向量$\theta^{(j)}$减去一个固定的向量$\psi$，这样参数就变成了$\theta^{(j)} - \psi$(j=1,...,k)，通过假设函数得到的各个分类的概率为：
$$\begin{align}
P(y^{(i)} = k | x^{(i)} ; \theta)
&= \frac{\exp((\theta^{(k)}-\psi)^\top x^{(i)})}{\sum_{j=1}^K \exp( (\theta^{(j)}-\psi)^\top x^{(i)})}  \\
&= \frac{\exp(\theta^{(k)\top} x^{(i)}) \exp(-\psi^\top x^{(i)})}{\sum_{j=1}^K \exp(\theta^{(j)\top} x^{(i)}) \exp(-\psi^\top x^{(i)})} \\
&= \frac{\exp(\theta^{(k)\top} x^{(i)})}{\sum_{j=1}^K \exp(\theta^{(j)\top} x^{(i)})}.
\end{align}$$
可以看到，减去一个固定向量和没减的结果是一样的！这表明了softmax回归的参数是“冗余”的。也可以说，softmax的模型是过度参数化的。存在多组参数值可以得到完全相同的假设函数。代价函数$J(\theta)$最小化的解$\theta$是不唯一的。

令$\psi = \theta^{(K)}$，可以令其减去一个固定的向量使得$\theta^{(K)} - \psi = \vec{0}$，而不影响假设函数。这样，我们就消去了参数向量$\theta^{(K)}$，现在只需要优化剩余的K-1 . n个参数。

###与逻辑回归的关系
令K=2，softmax回归就变成了logistic回归：
$$\begin{align}
h_\theta(x) &=

\frac{1}{ \exp(\theta^{(1)\top}x)  + \exp( \theta^{(2)\top} x^{(i)} ) }
\begin{bmatrix}
\exp( \theta^{(1)\top} x ) \\
\exp( \theta^{(2)\top} x )
\end{bmatrix}
\end{align}$$
由于softmax模型是过度参数化的，所以令其中一个参数向量为0，这样就变成：
$$\begin{align}
h(x) &=
\frac{1}{ \exp( (\theta^{(1)}-\theta^{(2)})^\top x^{(i)} ) + \exp(\vec{0}^\top x) }
\begin{bmatrix}
\exp( (\theta^{(1)}-\theta^{(2)})^\top x )
\exp( \vec{0}^\top x ) \\
\end{bmatrix} \\

&=
\begin{bmatrix}
\frac{1}{ 1 + \exp( (\theta^{(1)}-\theta^{(2)})^\top x^{(i)} ) } \\
\frac{\exp( (\theta^{(1)}-\theta^{(2)})^\top x )}{ 1 + \exp( (\theta^{(1)}-\theta^{(2)})^\top x^{(i)} ) }
\end{bmatrix} \\
&=
\begin{bmatrix}
\frac{1}{ 1  + \exp( (\theta^{(1)}-\theta^{(2)})^\top x^{(i)} ) } \\
1 - \frac{1}{ 1  + \exp( (\theta^{(1)}-\theta^{(2)})^\top x^{(i)} ) } \\
\end{bmatrix}
\end{align}$$

softmax回归估计一个类的概率为$\frac{1}{ 1  + \exp(- (\theta')^\top x^{(i)} ) }$，另一个类的概率为$1 - \frac{1}{ 1 + \exp(- (\theta')^\top x^{(i)} ) }$与logistic回归是相同的。

###练习
本次练习，我们将训练一个分类器可以对MNIST数据集中的10种数字进行分类。与logistic回归的练习只分类0和1两种不同的数字不同，本次练习会训练0到9一共10个数字。

1. 导入MNIST的训练数据和测试数据，并且增加一个截距系数
2. 使用logistic_regression.m作为目标函数调用minFunc。softmax_regression_vec.m 中计算目标函数$J(\theta; X,y)$的值并保存在f中，计算梯度$\nabla_\theta J(\theta; X,y)$的值并保存在g中。

```matlab
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
  g = g - X * (val_k - p)';
  g(:,end) = [];
  g=g(:); % make gradient a vector for minFunc
end
```
- theta是一个向量，需要先把它转换为 n x num_classes(10)的矩阵。而初始的theta只有num_classes-1列，需要增加一列即参数$\theta^{(K)}=0$
- 最后的返回的梯度值为 n x (num_classes-1)，将g的最后一列去掉

3. 最后运行代码，输出训练的准确率和测试的准确率
```
Reached Maximum Number of Iterations
Optimization took 36.675347 seconds.
Training accuracy: 94.4%
Test accuracy: 92.1%
```
通常，训练数据集的准确率是测试数据集的上界，即一般测试数据集的准确率不会高于训练数据集的准确率。

```matlab
addpath ../common
addpath ../common/minFunc_2012/minFunc
addpath ../common/minFunc_2012/minFunc/compiled

% Load the MNIST data for this exercise.
% train.X and test.X will contain the training and testing images.
%   Each matrix has size [n,m] where:
%      m is the number of examples.
%      n is the number of pixels in each image.
% train.y and test.y will contain the corresponding labels (0 to 9).
binary_digits = false;
num_classes = 10;
[train,test] = ex1_load_mnist(binary_digits);

% Add row of 1s to the dataset to act as an intercept term.
train.X = [ones(1,size(train.X,2)); train.X]; 
test.X = [ones(1,size(test.X,2)); test.X];
train.y = train.y+1; % make labels 1-based.
test.y = test.y+1; % make labels 1-based.

% Training set info
m=size(train.X,2);
n=size(train.X,1);

% Train softmax classifier using minFunc
options = struct('MaxIter', 200);

% Initialize theta.  We use a matrix where each column corresponds to a class,
% and each row is a classifier coefficient for that class.
% Inside minFunc, theta will be stretched out into a long vector (theta(:)).
% We only use num_classes-1 columns, since the last column is always
% assumed 0.（Remember ‘overparameterized'）
theta = rand(n,num_classes-1)*0.001;

% Call minFunc with the softmax_regression_vec.m file as objective.
%
% TODO:  Implement batch softmax regression in the softmax_regression_vec.m
% file using a vectorized implementation.
%
tic;
theta(:)=minFunc(@softmax_regression_vec, theta(:), options, train.X, train.y);
fprintf('Optimization took %f seconds.\n', toc);
theta=[theta, zeros(n,1)]; % expand theta to include the last class.

% Print out training accuracy.
tic;
accuracy = multi_classifier_accuracy(theta,train.X,train.y);
fprintf('Training accuracy: %2.1f%%\n', 100*accuracy);

% Print out test accuracy.
accuracy = multi_classifier_accuracy(theta,test.X,test.y);
fprintf('Test accuracy: %2.1f%%\n', 100*accuracy);


% % for learning curves
% global test
% global train
% test.err{end+1} = multi_classifier_accuracy(theta,test.X,test.y);
% train.err{end+1} = multi_classifier_accuracy(theta,train.X,train.y);
```

