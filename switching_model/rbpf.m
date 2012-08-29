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
    pts(ii).u(1) = (rand<params.prior_u);
    
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
        if rand < params.ppsl_switch
            pts(ii).u(kk) = 1-pts(ii).u(kk-1);
            switch_prob = log(params.p_switch/params.ppsl_switch);
        else
            pts(ii).u(kk) = pts(ii).u(kk-1);
            switch_prob = log((1-params.p_switch)/(1-params.ppsl_switch));
        end
        
        if pts(ii).u(kk) == 0
            var_z = params.var_z_0;
            var_y = params.var_y_0;
            A = params.A0;
        elseif pts(ii).u(kk) == 1
            var_z = params.var_z_1;
            var_y = params.var_y_1;
            A = params.A1;
        end
        
        % Kalman filter linear state
        [m_pred, P_pred] = kf_predict(pts(ii).m(:,kk-1), pts(ii).P(:,:,kk-1), A, var_z*params.Q);
        [m, P, ~, ~, ~, lhood] = kf_update(m_pred, P_pred, y(:,kk), params.C, var_y*params.R);
        P = (P+P')/2;
        assert(isposdef(P));
        pts(ii).m(:,kk) = m;
        pts(ii).P(:,:,kk) = P;
        
        % Calculate weight
        wts(ii) = wts(ii) + log(lhood) + switch_prob;
        
    end

% 
%     % Batch bootrap proposals for nonlinear states
%     pts_u = arrayfun(@(x) cat(1, x.u(kk-1)), pts);
%     new_pts_u = mvnrnd(pts_u, params.var_u);
%     
%     
%     % Extract moments from particles for current time k
%     pts_m = arrayfun(@(x) {x.m(:,kk)}, pts);
%     pts_P = arrayfun(@(x) {x.P(:,:,kk)}, pts);
%     
%     % Batch Kalman Filter
%     
%     
    
    % Normalise weights
    wts = wts - logsumexp(wts);
%     wts = wts - max(wts);
%     lin_wts = exp(wts); lin_wts = lin_wts/sum(lin_wts);
%     wts = log(lin_wts);
    
    % Store current state only
    bare_pts = pts;
    for ii = 1:Nf
        bare_pts(ii).m(:,[1:kk-1,kk+1:end]) = [];
        bare_pts(ii).P(:,:,[1:kk-1,kk+1:end]) = [];
        bare_pts(ii).u([1:kk-1,kk+1:end]) = [];
    end
    pts_array{kk} = bare_pts;
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

