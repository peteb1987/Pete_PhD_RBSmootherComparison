clup
dbstop if error
dbstop if warning

%% Set-up

batch = true;

if batch
    test_num = str2double(getenv('SGE_TASK_ID'));
%     test_num = 1;
else
    test_num = 1;
end

if batch
    params.d = test_num;
    params.K = 1000;
end

% Add toolbox folders to path
addpath('../toolbox/ekfukf/','../toolbox/arraylab/','../toolbox/lightspeed/','../toolbox/user/');

% DEFINE RANDOM SEED
rand_seed = 1;

% Set random seed
s = RandStream('mt19937ar', 'seed', rand_seed);
RandStream.setDefaultStream(s);

% Parameters
set_parameters;

%% Generate some data
[true_u, true_z, y] = generate_data(params);

%% Run RB particle filter
[filt_pts_array, filt_wts_array] = rbpf(params, y);

%% Run Kim's approx. smoother
[KA_smooth_pts] = rbps_KA(params, filt_pts_array, filt_wts_array, y);

%% Run linear-sampling smoother
[linsamp_smooth_pts] = rbps_linsamp(params, filt_pts_array, filt_wts_array, y);

%% Run full RB smoother
[full_smooth_pts] = rbps_full(params, filt_pts_array, filt_wts_array, y);

%% RMSEs

filt_smooth_u_est = mean(abs(cat(1, filt_pts_array{end}.u)));
KA_smooth_u_est = mean(abs(cat(1, KA_smooth_pts.u)));
linsamp_smooth_u_est = mean(abs(cat(1, linsamp_smooth_pts.u)));
full_smooth_u_est = mean(abs(cat(1, full_smooth_pts.u)));

filt_smooth_rmse = sqrt(mean( (filt_smooth_u_est - abs(true_u)).^2 ))
KA_smooth_rmse = sqrt(mean( (KA_smooth_u_est - abs(true_u)).^2 ))
linsamp_smooth_rmse = sqrt(mean( (linsamp_smooth_u_est - abs(true_u)).^2 ))
full_smooth_rmse = sqrt(mean( (full_smooth_u_est - abs(true_u)).^2 ))

if ~batch
    %% Data Plotting
    
    % Linear states
    for dd = 1:params.d
        figure, hold on
        plot(true_z(dd,:))
    end
    
    % Nonlinear state
    figure, hold on
    plot(true_u.^2, 'k', 'linewidth', 2);
    
    %% Plot Estimates
    plot(filt_smooth_u_est.^2, 'r')
    plot(KA_smooth_u_est.^2, 'm')
    plot(linsamp_smooth_u_est.^2, 'b')
    plot(full_smooth_u_est.^2, 'g')
    
else
    %% Save
    
    results.params = params;
    results.true_u = true_u;
    results.true_z = true_z;
    results.y = y;
    
    results.filt_smooth_u_est = filt_smooth_u_est;
    results.KA_smooth_u_est = KA_smooth_u_est;
    results.linsamp_smooth_u_est = linsamp_smooth_u_est;
    results.full_smooth_u_est = full_smooth_u_est;
    
    results.filt_smooth_rmse = filt_smooth_rmse;
    results.KA_smooth_rmse = KA_smooth_rmse;
    results.linsamp_smooth_rmse = linsamp_smooth_rmse;
    results.full_smooth_rmse = full_smooth_rmse;
    
    save(['alg_comparison_results' num2str(test_num)], 'results');
    
end