clear all
clc


% Input data : daily stocks price for 9 stocks
stocks = csvread('stockdata.csv');

% Computation of sample mean vector and sample covariance matrix of the
% returns

rets = stocks(2:end,:)./stocks(1:end-1,:)-1;

Sigma = cov(rets);
mu = mean(rets,1)';

one   = ones( size(mu) );
invSigma = inv(Sigma);

A = mu' * invSigma * one;
B = mu' * invSigma * mu;
C = one' * invSigma * one;
D = B*C - A^2;

% minimize the variance such that the weights sum to 1
f = zeros( size(mu) );
weights1 = quadprog(Sigma,f,[],[],one',1)

% minimize the variance such that mean daily return = .002

mustar = .002;
weights2 = quadprog(Sigma,f,-mu',-mustar,one',1)

% minimize the variance such that mean daily return = .002 and weights are
% positive
weights3 = quadprog(Sigma,f,-mu',-mustar,one',1, zeros( size(mu) ), one)

% fit a model that achieves a 1% daily return with all weights in [0,1]
mustar2 = .01;
weights4 = quadprog(Sigma,f,-mu',-mustar2,one',1, zeros(size(mu)), one) 


