function y = log_mvnpdf(X, Mu, Sigma)
%MVNPDF Log of Multivariate normal probability density function (pdf).

% USES COLUMN VECTORS FOR X and MU (as is sensible)
% NO ERROR CHECKING
% Can only handle a single covariance matrix

% Get size of data.
[d,n] = size(X);

% Offset mean
X0 = bsxfun(@minus, X, Mu);

% Cholesky decomposition
R = chol(Sigma);
xRinv = R' \ X0;
logSqrtDetSigma = sum(log(diag(R)));

% The quadratic form is the inner products of the standardized data
quadform = sum(xRinv.^2, 1);

y = -0.5*quadform - logSqrtDetSigma - d*log(2*pi)/2;
