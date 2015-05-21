# 逻辑回归
前一篇文章中，我们学习了利用线性回归（linear regression）来预测连续变量的值（如房价）。但是，有时候我们需要预测离散变量的值，如图片中的一个像素是“1”还是“0”。这就是一个**分类**的问题。**逻辑回归（logistic regression）**就是一个简单的算法。

在逻辑回归中，我们使用一个分类假设函数（hypothesis class）来预测给定输入属于类别“1”或者类别"0"的可能性。具体的，我们通过学习函数：
$$P(y=1|x)=h_\theta(x)=\frac{1}{1+exp(-\theta^Tx)}=\sigma(\theta^Tx),$$
$$P(y=0|x)=1-P(y=1|x)=1-h_\theta(x).$$
其中，
$$\sigma(z)=\frac{1}{1+exp(-z)}$$
称为逻辑函数（logistic function）或者sigmoid函数，其函数的形状如下所示，是一个取值范围在[0,1]之间的S型的曲线。
![sigmoid](http://i1.tietuku.com/e691dd6de68e01b2.png)

所以，我们可以把$$h_\theta(x)$$看成一个概率，我们的目标就是找到$$\theta$$使得当x属于类别"1"的时候$$P(y=1|x)=h_\theta(x)$$较大，而当x属于类别"0"的时候较小。

对于有m个样本的数据集，参数$$\theta$$的似然估计可以写为：$$L(\theta)=\prod_{i=1}^{m}p(y^{(i))}|x^{(i)};\theta)=\prod_{i=1}^{m}(h_\theta(x^{(i)}))^{y^{(i)}}(1-h_\theta(x^{(i)}))^{1-y^{(i))}}$$
定义损失函数：
$$J(\theta)=logL(\theta)=J(\theta) = - \sum_i \left(y^{(i)} \log( h_\theta(x^{(i)}) ) + (1 - y^{(i)}) \log( 1 - h_\theta(x^{(i)}) ) \right).$$

有了损失函数，我们可以通过训练数据找到参数$$\theta$$使得损失函数$$J(\theta)$$最小。等训练出分类模型之后，对于给定的输入，我们可以通过判断$$P(y=1|x) > P(y=0|x)$$，就把x标记为分类“1”，否则标记为分类“0”。

为了最小化损失函数，我们同样可以使用线性回归中使用的梯度下降法。因此，仍然需要计算每个$$\theta$$所对应的$$J(\theta)和\nabla_\theta J(\theta)$$。对于给定的$$\theta_j，J(\theta)$$的梯度可以表示为：
$$\nabla_\theta J(\theta) = \sum_i x^{(i)} (h_\theta(x^{(i)}) - y^{(i)})$$
这与线性回归的梯度的形式一样，只是这里的$$h_\theta(x) = \sigma(\theta^\top x).$$


## 练习
这个练习主要是使用逻辑回归来对[MNIST dataset](http://yann.lecun.com/exdb/mnist/)的数字图片进行分类，其中我们使用的数据中只有数字“1”或“0”两种。例如，这些数字图片如下所示：
![mnist](http://ufldl.stanford.edu/tutorial/images/Mnist_01.png)

每一个数字用一个28x28的像素构成，可以用一个28x28 = 784个元素的向量$$x^{(i)}$$表示，而分类的标签只有两种$$y^{(i)} \in \{0,1\}$$。


1.使用ex1_load_mnist.m导入MNIST的训练数据和测试数据。矩阵X为每一个的值，即$$X_{ji}=x_j^{(j)}$$表示第i个样本的第j个像素值，向量y是类别标签。另外，将X进行标准化，使其每一行的标准差为1，均值0.

```matlab
% ex1_load_mnist.m
function [train, test] = ex1_load_mnist(binary_digits)

  % Load the training data
  X=loadMNISTImages('train-images-idx3-ubyte');
  y=loadMNISTLabels('train-labels-idx1-ubyte')';

  if (binary_digits)
    % Take only the 0 and 1 digits
    X = [ X(:,y==0), X(:,y==1) ];
    y = [ y(y==0), y(y==1) ];
  end

  % Randomly shuffle the data
  I = randperm(length(y));
  y=y(I); % labels in range 1 to 10
  X=X(:,I);

  % We standardize the data so that each pixel will have roughly zero mean and unit variance.
  s=std(X,[],2);
  m=mean(X,2);
  X=bsxfun(@minus, X, m);
  X=bsxfun(@rdivide, X, s+.1);

  % Place these in the training set
  train.X = X;
  train.y = y;

  % Load the testing data
  X=loadMNISTImages('t10k-images-idx3-ubyte');
  y=loadMNISTLabels('t10k-labels-idx1-ubyte')';

  if (binary_digits)
    % Take only the 0 and 1 digits
    X = [ X(:,y==0), X(:,y==1) ];
    y = [ y(y==0), y(y==1) ];
  end

  % Randomly shuffle the data
  I = randperm(length(y));
  y=y(I); % labels in range 1 to 10
  X=X(:,I);

  % Standardize using the same mean and scale as the training data.
  X=bsxfun(@minus, X, m);
  X=bsxfun(@rdivide, X, s+.1);

  % Place these in the testing set
  test.X=X;
  test.y=y;
```

2.在数据里增加一行数值全部为1的数据，作为截距的系数。


3.使用logistic_regression.m作为目标函数调用minFunc
```matlab
% logistic_regression.m
function [f,g] = logistic_regression(theta, X,y)
  %
  % Arguments:
  %   theta - A column vector containing the parameter values to optimize.
  %   X - The examples stored in a matrix.
  %       X(i,j) is the i'th coordinate of the j'th example.
  %
  %   y - The label for each example.  y(j) is the j'th example's label.
  %

  m=size(X,2);

  % initialize objective value and gradient.
  f = 0;
  g = zeros(size(theta));


  %
  % TODO:  Compute the objective function by looping over the dataset and summing
  %        up the objective values for each example.  Store the result in 'f'.
  %
  % TODO:  Compute the gradient of the objective by looping over the dataset and summing
  %        up the gradients (df/dtheta) for each example. Store the result in 'g'.
  %

 for j=1:m
     f = f - (y(j) * log(sigmoid(theta' * X(:,j))) + (1 - y(j)) * log(1 - sigmoid(theta' * X(:,j))));
     g = g + X(:,j) * (sigmoid(theta' * X(:,j)) - y(j));
 end
end
```
返回目标函数及其梯度的值。


4.miniFunc执行结束后，输出执行的时间、训练的准确度和测试的准确度。
```
Step Size below progTol
Optimization took 72.327744 seconds.
Training accuracy: 100.0%
Test accuracy: 100.0%
```

可以看到由于只需要对“0”和“1”进行分类，这两个数字比较容易区分开，所以分类的准确度非常高，达到了100%。

整个过程的代码如下：
```matlab
addpath ../common
addpath ../common/minFunc_2012/minFunc
addpath ../common/minFunc_2012/minFunc/compiled
addpath ../common/data

% Load the MNIST data for this exercise.
% train.X and test.X will contain the training and testing images.
%   Each matrix has size [n,m] where:
%      m is the number of examples.
%      n is the number of pixels in each image.
% train.y and test.y will contain the corresponding labels (0 or 1).
binary_digits = true;
[train,test] = ex1_load_mnist(binary_digits);

% Add row of 1s to the dataset to act as an intercept term.
train.X = [ones(1,size(train.X,2)); train.X];
test.X = [ones(1,size(test.X,2)); test.X];

% Training set dimensions
m=size(train.X,2);
n=size(train.X,1);

% Train logistic regression classifier using minFunc
options = struct('MaxIter', 100);

% First, we initialize theta to some small random values.
theta = rand(n,1)*0.001;

% Call minFunc with the logistic_regression.m file as the objective function.
%
% TODO:  Implement batch logistic regression in the logistic_regression.m file!
%
tic;
theta=minFunc(@logistic_regression, theta, options, train.X, train.y);
fprintf('Optimization took %f seconds.\n', toc);

% Now, call minFunc again with logistic_regression_vec.m as objective.
%
% TODO:  Implement batch logistic regression in logistic_regression_vec.m using
% MATLAB's vectorization features to speed up your code.  Compare the running
% time for your logistic_regression.m and logistic_regression_vec.m implementations.
%
% Uncomment the lines below to run your vectorized code.
%theta = rand(n,1)*0.001;
%tic;
%theta=minFunc(@logistic_regression_vec, theta, options, train.X, train.y);
%fprintf('Optimization took %f seconds.\n', toc);

% Print out training accuracy.
tic;
accuracy = binary_classifier_accuracy(theta,train.X,train.y);
fprintf('Training accuracy: %2.1f%%\n', 100*accuracy);

% Print out accuracy on the test set.
accuracy = binary_classifier_accuracy(theta,test.X,test.y);
fprintf('Test accuracy: %2.1f%%\n', 100*accuracy);

```











