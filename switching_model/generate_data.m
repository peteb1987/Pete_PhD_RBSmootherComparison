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
y = zeros(size(params.C,1),K);

% Sample priors
u(1) = (rand<params.prior_u);
z(:,1) = mvnrnd(zeros(d,1), params.prior_var_z)';
y(1) = NaN;

% Loop through time
for kk = 2:K
    
    % Propagate u
    if rand < params.p_switch
        u(kk) = 1-u(kk-1);
    else
        u(kk) = u(kk-1);
    end
    
    % Propagate z
    if u(kk) == 0
        var_z = params.var_z_0;
        var_y = params.var_y_0;
        A = params.A0;
    elseif u(kk) == 1
        var_z = params.var_z_1;
        var_y = params.var_y_1;
        A = params.A1;
    else
        error('Invalid model choice');
    end
    
    z(:,kk) = mvnrnd(A*z(:,kk-1), var_z*params.Q);
    
    % Observe
    y(:,kk) = mvnrnd(params.C*z(:,kk), var_y*params.R);
    
end

end

