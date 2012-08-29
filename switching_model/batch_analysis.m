clup

num_test = 100;
test_per_d = 10;
save_flag = false;
addpath('./batch_M1_1dnoise_lowtransprob_K1000_Nf100_Ns10/');

num_d = ceil(num_test/test_per_d);
load(['alg_comparison_results' num2str(1)]);

u_err_rate = cell(4,1);
u_av_err_rate = cell(4,1);
u_pred_rate = cell(4,1);
u_av_pred_rate = cell(4,1);
z_mae = cell(4,1);
z1_av_mae = cell(4,1);

% Loop through algorithms
for alg = 1:4
    if isempty(u_err_rate{alg}), u_err_rate{alg} = cell(num_d,1); end
    if isempty(u_pred_rate{alg}), u_pred_rate{alg} = cell(num_d,1); end
    if isempty(z_mae{alg}), z_mae{alg} = cell(num_d,1); end
    
    % Loop through tests collating results
    for ii = 1:num_test
        load(['alg_comparison_results' num2str(ii)]);
        d = results.params.d;
        
        if isempty(u_err_rate{alg}{d}), u_err_rate{alg}{d} = zeros(1,results.params.K); end
        if isempty(u_pred_rate{alg}{d}), u_pred_rate{alg}{d} = zeros(1,results.params.K); end
        if isempty(z_mae{alg}{d}), z_mae{alg}{d} = zeros(d,results.params.K); end
        
        if alg == 1
            num_pts = results.params.Nf;
        else
            num_pts = results.params.Ns;
        end
        
        u_err_rate{alg}{d} = u_err_rate{alg}{d} + (results.u_est{alg} ~= results.true_u);
        u_pred_rate{alg}{d} = u_pred_rate{alg}{d} + (num_pts-results.mode_pts{alg})/num_pts;
        z_mae{alg}{d} = z_mae{alg}{d} + results.z_err{alg};
    end
    for d = 1:num_d
        u_err_rate{alg}{d} = u_err_rate{alg}{d}/test_per_d;
        u_pred_rate{alg}{d} = u_pred_rate{alg}{d}/test_per_d;
        z_mae{alg}{d} = z_mae{alg}{d}/test_per_d;
        tmp = mean(z_mae{alg}{d},2);
        z1_av_mae{alg}(d) = tmp(1);
    end
    u_av_err_rate{alg} = mean(cell2mat(u_err_rate{alg}),2);
    u_av_pred_rate{alg} = mean(cell2mat(u_pred_rate{alg}),2);
    
end

figure, hold on,
plot([1:num_d], u_av_err_rate{1}, 'r')
plot([1:num_d], u_av_err_rate{2}, 'm')
plot([1:num_d], u_av_err_rate{3}, 'b')
plot([1:num_d], u_av_err_rate{4}, 'g')
xlabel('d')
ylabel('Error rate')
legend('Filter-smoother', 'Kim''s approximation', 'Linear sampling', 'Full RB');

figure, hold on,
plot([1:num_d], u_av_pred_rate{1}, 'r')
plot([1:num_d], u_av_pred_rate{2}, 'm')
plot([1:num_d], u_av_pred_rate{3}, 'b')
plot([1:num_d], u_av_pred_rate{4}, 'g')
xlabel('d')
ylabel('Predicted error rate')
legend('Filter-smoother', 'Kim''s approximation', 'Linear sampling', 'Full RB');

figure, hold on,
plot([1:num_d], z1_av_mae{1}(1,:), 'r')
plot([1:num_d], z1_av_mae{2}(1,:), 'm')
plot([1:num_d], z1_av_mae{3}(1,:), 'b')
plot([1:num_d], z1_av_mae{4}(1,:), 'g')
xlabel('d')
ylabel('Linear mean absolute error')
legend('Filter-smoother', 'Kim''s approximation', 'Linear sampling', 'Full RB');

if save_flag
    export_pdf(gcf, 'err_rate_comparison');
end