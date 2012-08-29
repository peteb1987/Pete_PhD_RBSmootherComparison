%BATCH_KF_PREDICT  Perform Kalman Filter prediction steps with a batch of
%Kalman filters, with inputs of each in cell arrays.
%
% Syntax:
%   [X,P] = KF_PREDICT(X,P,A,Q)
%
%       Each of the following contained in a Mx1 or 1xM cell array, or a
%       single cell for all in batch
% In:
%   X - Nx1 mean state estimate of previous step
%   P - NxN state covariance of previous step
%   A - Transition matrix of discrete model (optional, default identity)
%   Q - Process noise of discrete model     (optional, default zero)
%
% Out:
%   X - Predicted state mean
%   P - Predicted state covariance
%   
% Description:
%   Perform Kalman Filter prediction step. The model is
%
%     x[k] = A*x[k-1] + q,  q ~ N(0,Q).
% 
%   The predicted state is distributed as follows:
%   
%     p(x[k] | x[k-1]) = N(x[k] | A*x[k-1], Q[k-1])
%
%   The predicted mean x-[k] and covariance P-[k] are calculated
%   with the following equations:
%
%     m-[k] = A*x[k-1]
%     P-[k] = A*P[k-1]*A' + Q.
%

function [x,P] = batch_kf_predict(x,P,A,Q)

% Number in batch
N = size(x,1);
d = size(x{1},1);

%
% Apply defaults
%
if (nargin<3)||isempty(A)
    A = repmat({eye(d)}, N, 1);
end
if (nargin<4)||isempty(Q)
    Q = repmat({zeros(d)}, N, 1);
end

%
% Perform prediction
%
x = cellfun(@(Ai,xi) {Ai*xi}, A, x);
P = cellfun(@(Ai,Qi,Pi) {Ai*Pi*Ai'+Qi}, A, Q, P);

end

