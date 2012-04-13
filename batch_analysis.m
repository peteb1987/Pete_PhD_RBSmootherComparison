addpath('./Batch_Nf100_Ns100/');

filt_smooth_array = zeros(10,1); for ii=1:10, load(['alg_comparison_results' num2str(ii)]), filt_smooth_array(ii) = results.filt_smooth_rmse; end
KA_smooth_array = zeros(10,1); for ii=1:10, load(['alg_comparison_results' num2str(ii)]), KA_smooth_array(ii) = results.KA_smooth_rmse; end
linsamp_smooth_array = zeros(10,1); for ii=1:10, load(['alg_comparison_results' num2str(ii)]), linsamp_smooth_array(ii) = results.linsamp_smooth_rmse; end
full_smooth_array = zeros(10,1); for ii=1:10, load(['alg_comparison_results' num2str(ii)]), full_smooth_array(ii) = results.full_smooth_rmse; end

figure, hold on,
plot([1:10], filt_smooth_array, 'r')
plot([1:10], KA_smooth_array, 'm')
plot([1:10], linsamp_smooth_array, 'b')
plot([1:10], full_smooth_array, 'g')
xlabel('d')
ylabel('RMSE')
export_pdf(gcf, 'RMSE_comparison');