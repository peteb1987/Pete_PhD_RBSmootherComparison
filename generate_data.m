function [ u, z, y ] = generate_data( params )
%GENERATE_DATA Generate data from a linear-Gaussian conditionally
%linear-Gaussian model.

% u - nonlinear state
% z - linear state
% y - observations

K = params.K;
d = params.d;

% Set up arrays
u = zeros(1,K);
z = zeros(d,K);
y = zeros(1,K);

% Sample priors
u(1) = mvnrnd(0, params.prior_var_u)';
z(:,1) = mvnrnd(zeros(d,1), params.prior_var_z)';
y(1) = NaN;

% Loop through time
for kk = 2:K
    
    % Propagate u
    u(1,kk) = mvnrnd(params.alpha*u(kk-1), params.var_u);
    
    % Propagate z
    z(:,kk) = mvnrnd(params.A*z(:,kk-1), u(kk)^2*params.Q);
    
    % Observe
    y(1,kk) = mvnrnd(z(1,kk), params.var_y);
    
end

end

