function [pts_array, wts_array] = rbpf(params, y)
%RBPF Run a RB particle filter

K = params.K;
d = params.d;
Nf = params.Nf;

% Initialise particle array
pts = struct('u', repmat({zeros(1,K)}, Nf, 1), 'm', repmat({zeros(d,K)}, Nf, 1), 'P', repmat({zeros(d,d,K)}, Nf, 1));
wts = log(ones(Nf,1)/Nf);

% Storage arrays
pts_array = cell(K,1);
wts_array = cell(K,1);
pts_array{1} = pts;
wts_array{1} = wts;

% Priors
for ii = 1:Nf
    
    % Sample nonlinear part
    pts(ii).u(1) = mvnrnd(0, params.prior_var_u);
    
    % Set moments for linear part
    pts(ii).m(:,1) = zeros(d,1);
    pts(ii).P(:,:,1) = params.prior_var_z;
    
end

% Loop through time
for kk = 2:K
    
    fprintf(1, '*** Filtering time step %u.\n', kk);
    
    % Loop through particles
    for ii = 1:Nf
        
        % Bootstrap proposal for nonlinear state
        pts(ii).u(kk) = mvnrnd(params.alpha*pts(ii).u(kk-1), params.var_u);
        
        % Kalman filter linear state
        [m_pred, P_pred] = kf_predict(pts(ii).m(:,kk-1), pts(ii).P(:,:,kk-1), params.A, (pts(ii).u(kk).^2)*params.Q);
        [pts(ii).m(:,kk), pts(ii).P(:,:,kk), ~, ~, ~, lhood] = kf_update(m_pred, P_pred, y(kk), params.C, params.var_y);
        pts(ii).P(:,:,kk) = (pts(ii).P(:,:,kk)+pts(ii).P(:,:,kk)')/2;
        
        % Calculate weight
        wts(ii) = wts(ii) + log(lhood);
        
    end
    
    % Normalise weights
    wts = wts - logsumexp(wts);
%     wts = wts - max(wts);
%     lin_wts = exp(wts); lin_wts = lin_wts/sum(lin_wts);
%     wts = log(lin_wts);
    
    % Store
    pts_array{kk} = pts;
    wts_array{kk} = wts;
    
    % Resample
    [~, parents] = systematic_resample(exp(wts), Nf);
    pts = pts(parents);
    wts = log(ones(Nf,1)/Nf);
    
    if kk == K
        % For final frame, store resampled version
        pts_array{kk} = pts;
        wts_array{kk} = wts;
    end
    
end

end

