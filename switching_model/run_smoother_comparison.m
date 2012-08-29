close all
clear all
clc

% Algorithms
% 1 - filter smoother
% 2 - Kim's approximation smoother
% 3 - Linear sampling smoother
% 4 - Full RB smoother

%% Set-up

batch = false;

if batch
    test_num = str2double(getenv('SGE_TASK_ID'));
%     test_num = 1;
    test_per_d = 10;
    
    params.d = floor((test_num-1)/test_per_d)+1;
    
    params.K = 1000;
    params.Nf = 100;
    params.Ns = 10;
    
    % Add toolbox folders to path
    addpath('../toolbox/ekfukf/','../toolbox/arraylab/','../toolbox/lightspeed/','../toolbox/user/');
    
    % DEFINE RANDOM SEED
    rand_seed = test_num;
    
else
    test_num = 1;
    
    params.d = 5;
    params.K = 1000;
    params.Nf = 100;
    params.Ns = 10;
    
    dbstop if error
    dbstop if warning
    
    % DEFINE RANDOM SEED
    rand_seed = 1;
    
end

% Set random seed
s = RandStream('mt19937ar', 'seed', rand_seed);
RandStream.setDefaultStream(s);

% Parameters
set_parameters;

%% Generate some data
[true_u, true_z, y] = generate_data(params);

%% Create arrays
smooth_pts = cell(4,1);

%% Run RB particle filter
[filt_pts_array, filt_wts_array] = rbpf(params, y);
smooth_pts{1} = filt_pts_array{end};

%% Run Kim's approx. smoother
[smooth_pts{2}] = rbps_KA(params, filt_pts_array, filt_wts_array, y);

%% Run linear-sampling smoother
[smooth_pts{3}] = rbps_linsamp(params, filt_pts_array, filt_wts_array, y);

%% Run full RB smoother
[smooth_pts{4}] = rbps_full(params, filt_pts_array, filt_wts_array, y);

%% Diagnostics

u_est = cell(4,1);          % Modal indicator variable over time
mode_pts = cell(4,1);       % Number of particles selecting the mode over time
z_err = cell(4,1);          % Linear state error

% Loop through results
for alg = 1:4
    
    % How many particles?
    num_pts = length(smooth_pts{alg});
    
    % Choose mode and count particles
    [u_est{alg}, mode_pts{alg}] = mode(cat(1, smooth_pts{alg}.u));
    
    % Find linear state estimate
    z_est = mean(cell2mat(permute(arrayfun(@(x) {x.m}, smooth_pts{alg}),[3 2 1])),3);
    
    % Calculate linear state rmse
    z_err{alg} = abs(z_est - true_z);
end

if ~batch
    %% Output
    
    mean(abs(true_u-u_est{1}))
    mean(abs(true_u-u_est{2}))
    mean(abs(true_u-u_est{3}))
    mean(abs(true_u-u_est{4}))
    
    mean(z_err{1}(1,:))
    mean(z_err{2}(1,:))
    mean(z_err{3}(1,:))
    mean(z_err{4}(1,:))
    
    %% Data Plotting
    
    % Linear states
    if params.d < 25
        for dd = 1:params.d
            figure, hold on
            plot(true_z(dd,:), 'b')
        end
    end
    
    % Nonlinear state
    figure, hold on
    plot(true_u.^2, 'k', 'linewidth', 2);
    
    %% Plot Estimates
    plot(u_est{1}, 'r')
    plot(u_est{2}, 'm')
    plot(u_est{3}, 'b')
    plot(u_est{4}, 'g')
    legend('True', 'Filter-smoother', 'Kim''s approximation', 'Linear sampling', 'Full RB');
    
else
    %% Save
    
    results.params = params;
    results.true_u = true_u;
    results.true_z = true_z;
    results.y = y;
    
    results.u_est = u_est;
    results.mode_pts = mode_pts;
    results.z_err = z_err;
    
    save(['alg_comparison_results' num2str(test_num)], 'results');
    
end