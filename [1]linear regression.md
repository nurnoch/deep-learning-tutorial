# 线性回归
首先，学习如何实现线性回归。包括：
- 熟悉目标函数
- 计算梯度
- 通过一系列的参数优化目标函数
这些基础的算法和工具对于后面学习复杂的算法非常重要。

## 问题描述
线性回归的目标就是：通过输入向量$$x \in \Re^n$$，来预测目标值$$y$$。例如，我们想预测房价y，那么x就是一系列描述房子的特征（如房子大小、卧室的数目等）。

我们的目标就是找到一个函数$$y=h(x)$$，使得对于每一个训练样本有$$y^{(i)} \approx h(x^{i})$$。这样我们就可以$$h(x)$$来预测新的输入变量$$x$$。

为了找到函数$$h(x)$$，我们必须决定如何表示它。现在，我们使用线性函数：
$$h_\theta(x)=\sum_j\theta_jx_j=\theta^Tx$$
因此，我们现在的任务就是要找到参数$$\theta$$使$$h_\theta(x^{(i)})$$与$$y^{(i)}$$尽可能接近。即**通过选择参数$$\theta$$最小化损失函数**：

$$J(\theta) = \frac{1}{2} \sum_i \left( h_\theta(x^{(i)}) - y^{(i)} \right)^2 = \frac{1}{2} \sum_i \left( \theta^\top x^{(i)} - y^{(i)} \right)^2$$


## 损失函数最小化
为了选择一个$$\theta$$使得$$J(\theta)$$最小。我们可以首先选择$$\theta$$的初始值，然后重复的改变$$\theta$$使得$$J(\theta)$$更小，直到$$J(\theta)$$收敛到一个最小值。有很多求函数最小值的算法，其中常见的一种就是[梯度下降法](http://www.wengweitao.com/ti-du-xia-jiang-fa.html)。目前，我们暂且先认为求解函数最小值的方法需要计算每一个参数$$\theta$$对应的：

损失函数：
$$J(\theta)$$
损失函数的梯度：
$$\nabla_\theta J(\theta)$$
然后，利用优化算法找到最优的参数$$\theta$$.

损失函数可以将所有的样本数据$$(x^{(i)}, y^{(i)})$$代入损失函数的表达式中计算即可。而梯度：
$$\nabla_\theta J(\theta) = \begin{align}\left[\begin{array}{c} \frac{\partial J(\theta)}{\partial \theta_1}  \\
\frac{\partial J(\theta)}{\partial \theta_2}  \\
\vdots\\
\frac{\partial J(\theta)}{\partial \theta_n} \end{array}\right]\end{align}$$
其中：
$$\frac{\partial J(\theta)}{\partial \theta_j} = \frac{\partial \frac{1}{2}(h_\theta(x)-y)^2}{\partial \theta_j} \\
=(h_\theta(x)-y) \frac{\partial (\sum_i \theta_ix_i-y)}{\partial \theta_j}\\
 =  \sum_i x^{(i)}_j \left(h_\theta(x^{(i)}) - y^{(i)}\right)$$


##练习
在这个练习中，我们会实现线性回归的目标函数以及梯度的计算。在ex1/文件夹中包含了起始的代码和本练习中会用到的数据。在文件 ex1_linreg.m 中已经列出了主要的步骤和样本。

###数据
1.数据保存在 housing.data 文件中。初始的数据包括506行14列。

2.首先，需要将数据进行转置，将原来的行变为列，列变为行，即转置后的数据包含14行506列。

3.在数据的开头增加一行数值全部为1的数据，这样$$\theta_1$$就可以用来表示线性回归中的截距。现在的数据包含15行506列，每一列代表一个样本数据。第1-13行为输入特征x，最后一行为要预测的值y。

4.将数据列的排列顺序随机打乱

5.将数据分成训练数据集和测试数据集两个部分。训练数据集用来训练出参数$$\theta$$，而测试数据集用来预测新的输入x，并检验预测函数的效果。

###代码实现
1.首先，按以上的步骤对数据进行处理，并将数据分为训练数据和测试数据；
```matlab
% Load housing data from file.
data = load('housing.data');
data=data'; % put examples in columns

% Include a row of 1s as an additional intercept feature.
data = [ ones(1,size(data,2)); data ];

%  examples.
data = data(:, randperm(size(data,2)));

% The number of row
% rows = size(data,1);

% Split into train and test sets
% The last row of 'data' is the median home price.
train.X = data(1:end-1, 1:400);
train.y = data(end,1:400);

test.X = data(1:end-1,401:end);
test.y = data(end,401:end);
```

2.代码会调用一个称为 miniFunc 的优化的Matlab包。miniFunc会寻找最小化目标函数的参数$$\theta$$
```matlab
addpath ../common
addpath ../common/minFunc_2012/minFunc
addpath ../common/minFunc_2012/minFunc/compiled



% m is the number of samples
m=size(train.X,2);
% n is the number of features
n=size(train.X,1);

% Initialize the coefficient vector theta to random values.
theta = rand(n,1);

% Run the minFunc optimizer with linear_regression.m as the objective.
%
% TODO:  Implement the linear regression objective and gradient computations
% in linear_regression.m
%
tic;
options = struct('MaxIter', 200);
theta = minFunc(@linear_regression, theta, options, train.X, train.y);
fprintf('Optimization took %f seconds.\n', toc);
```

3.实现linear_regression，计算出当前参数$$\theta$$所对应的损失函数值和梯度分别保存在f和g中。
```matalb
% linear_regression.m
function [f,g] = linear_regression(theta, X,y)
  %
  % Arguments:
  %   theta - A vector containing the parameter values to optimize.
  %   X - The examples stored in a matrix.
  %       X(i,j) is the i'th coordinate of the j'th example.
  %   y - The target value for each example.  y(j) is the target for example j.
  %

  m=size(X,2);
  n=size(X,1);

  f=0;
  g=zeros(size(theta));

  %
  % TODO:  Compute the linear regression objective by looping over the examples in X.
  %        Store the objective function value in 'f'.
  %
  % TODO:  Compute the gradient of the objective with respect to theta by looping over
  %        the examples in X and adding up the gradient for each example.  Store the
  %        computed gradient in 'g'.

%%% YOUR CODE HERE %%%
for j = 1:m
  f = f + 1/2 * (theta' * X(:,j) - y(j))^2;
  g = g + X(:,j) * (theta' * X(:,j) - y(j));
end
end
```

4.可以输出训练数据和测试数据的均方根（RMS）误差。
```matlab
% Plot predicted prices and actual prices from training set.
actual_prices = train.y;
predicted_prices = theta'*train.X;

% Print out root-mean-squared (RMS) training error.
train_rms=sqrt(mean((predicted_prices - actual_prices).^2));
fprintf('RMS training error: %f\n', train_rms);

% Print out test RMS error
actual_prices = test.y;
predicted_prices = theta'*test.X;
test_rms=sqrt(mean((predicted_prices - actual_prices).^2));
fprintf('RMS testing error: %f\n', test_rms);
```
均方根误差又称为标准误差，是观测值与真实值的平方和观测次数n比值的平方根。

5.最后可以画出在测试数据上预测值和真实值的分布图
```matlab
% Plot predictions on test data.
plot_prices=true;
if (plot_prices)
  [actual_prices,I] = sort(actual_prices);
  predicted_prices=predicted_prices(I);
  plot(actual_prices, 'rx');
  hold on;
  plot(predicted_prices,'bx');
  legend('Actual Price', 'Predicted Price');
  xlabel('House #');
  ylabel('House price ($1000s)');
end
```

![linear_regression](http://i1.tietuku.com/89d7211831ec77dd.jpg)









