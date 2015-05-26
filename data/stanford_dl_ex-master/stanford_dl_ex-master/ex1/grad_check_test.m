%
%This exercise uses a data from the UCI repository:
% Bache, K. & Lichman, M. (2013). UCI Machine Learning Repository
% http://archive.ics.uci.edu/ml
% Irvine, CA: University of California, School of Information and Computer Science.
%
%Data created by:
% Harrison, D. and Rubinfeld, D.L.
% ''Hedonic prices and the demand for clean air''
% J. Environ. Economics & Management, vol.5, 81-102, 1978.
%
addpath ../common
addpath ../common/minFunc_2012/minFunc
addpath ../common/minFunc_2012/minFunc/compiled

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

avg_error = grad_check(@linear_regression, theta, 10, train.X, train.y)