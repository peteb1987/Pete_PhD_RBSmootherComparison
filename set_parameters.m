% Model
% params.K = 1000;                     % Number of time steps
% params.d = 3;                      % Number of dimensions
                d = params.d;
params.dt = 1;                      % Time step

params.var_y = 0.01;                   % Observation variance
params.var_u = 0.01;                % Nonlinear state transition variance

params.prior_var_u = 0.01;          % Nonlinear state prior variance
params.prior_var_z = 0.1*eye(d);    % Linear state prior covariance

params.alpha = 0.99;                 % Decay constant for nonlinear state
params.lambda = 0.9.^flipud(cumsum(ones(d,1)));      % Decay constants for linear state


% Calculate linear state transition matrices (Q up to scale factor sigma^2)
F = [zeros(d-1,1), eye(d-1); zeros(1,d)] - diag(params.lambda);
L = [zeros(d-1,1); 1];
Qc = 1;
[params.A, params.Q] = lti_disc(F, L, Qc, params.dt);
params.invA = inv(params.A);
params.invQ = params.A\params.Q/params.A';
params.C = [1, zeros(1,d-1)];

% Algorithms
params.Nf = 100;
params.Ns = 10;
