% Model
% params.K = 1000;                     % Number of time steps
% params.d = 3;                      % Number of dimensions
                d = params.d;
params.dt = 1;                      % Time step

params.var_y = 1;                   % Observation variance
params.p_switch = 0.001;             % Model switching probability
params.var_z_0 = 10;                % Model 0 variance
params.var_z_1 = 0.1;               % Model 1 variance

params.ppsl_switch = 0.1;           % Proposal probability for model switch (not too low!)

params.var_y_0 = 1;
params.var_y_1 = 1;

params.prior_u = 0.5;               % Model 0 prior probability
params.prior_var_z = 1*eye(d);    % Linear state prior covariance

params.lambda = 0.9.^flipud(cumsum(ones(d,1)));      % Decay constants for linear state
% params.lambda = 0.9;1;


% Calculate linear state transition matrices (Q up to scale factor sigma^2)

% Model 1 - integrator chain
F = [zeros(d-1,1), eye(d-1); zeros(1,d)] - diag(params.lambda);
L = eye(d);%[zeros(d-1,1); 1];%
params.C = [1, zeros(1,d-1)];
params.R = 1;

% % Model 2 - parallel drift-diffusions
% assert(rem(d,2)==0);
% Fi = [0 1; 0 -params.lambda];
% Fcopy = cell(d/2,1);
% [Fcopy{:}] = deal(Fi);
% F = blkdiag(Fcopy{:});
% L = zeros(d, d/2);
% for ii = 1:d/2, L(2*ii,ii) = 1; end
% params.C = zeros(d/2, d);
% for ii = 1:d/2, params.C(ii,2*ii-1) = 1; end
% params.R = params.var_y * eye(d/2);

% Model 3 - parallel BMs
% F = zeros(d);
% L = eye(d);
% params.C = eye(d);
% params.R = params.var_y * eye(d);

Qc = 1;
[params.A, params.Q] = lti_disc(F, L, Qc, params.dt);
params.Q = (params.Q+params.Q')/2;
params.invA = inv(params.A);
params.invQ = params.A\params.Q/params.A';

% params.A0 = eye(d);
params.A0 = params.A;
params.A1 = params.A;

% Algorithms
% params.Nf = 100;
% params.Ns = 100;
