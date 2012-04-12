function [ pts ] = rbps_full( params, filt_pts_array, filt_wts_array, y )
%RBPS_FULL Run a complete RB particle smoother

d = params.d;
K = params.K;
Nf = params.Nf;
Ns = params.Ns;

% Initialise particle array
pts = struct('u', repmat({zeros(1,K)}, Ns, 1), 'm', repmat({zeros(d,K)}, Ns, 1), 'P', repmat({zeros(d,d,K)}, Ns, 1));

% Loop through particles
for ii = 1:Ns
    
    fprintf(1, '*** Smoothing particle %u.\n', ii);
    
    % Initialise particle by sampling from final filtering frame
    ind = randsample(Nf, 1, true, exp(filt_wts_array{K}));
    pts(ii).u(K) = filt_pts_array{K}(ind).u(K);
    
    % Initialise backwards filter (with approximation here)
    m_bwds = filt_pts_array{K}(ind).m(:,K);
    P_bwds = filt_pts_array{K}(ind).P(:,:,K);
    
    % Loop backwards through time
    for kk = K-1:-1:2
        
        % Run the backwards KF prediction
        [m_bwds, P_bwds] = kf_predict(m_bwds, P_bwds, params.invA, (pts(ii).u(kk+1)^2)*params.invQ);
        P_bwds = (P_bwds+P_bwds')/2;
        assert(isposdef(P_bwds))
        
        % Initialise array of sampling weights
        samp_wts = zeros(Nf,1);
        
        % Loop through filter particles
        for jj = 1:Nf
            
            % Nonlinear bit
            nonlin_prob = log(mvnpdf(pts(ii).u(kk+1), params.alpha*filt_pts_array{kk}(jj).u(kk), params.var_u));
            
            % Linear bit
            lin_prob = log(mvnpdf(filt_pts_array{kk}(jj).m(:,kk), m_bwds, filt_pts_array{kk}(jj).P(:,:,kk)+P_bwds));
            
            % Calculate sampling weight
            samp_wts(jj) = filt_wts_array{kk}(jj) + lin_prob + nonlin_prob;
            
        end
        
        % Normalise weights
        samp_wts = samp_wts - logsumexp(samp_wts);
        
        % Sample
        ind = randsample(Nf, 1, true, exp(samp_wts));
        pts(ii).u(kk) = filt_pts_array{kk}(ind).u(kk);
        
        % Run the backwards KF update
        [m_bwds, P_bwds] = kf_update(m_bwds, P_bwds, y(kk), params.C, params.var_y);
        P_bwds = (P_bwds+P_bwds')/2;
        assert(isposdef(P_bwds))
        
    end
    
    % Run a RTS smoother to ascertain the linear parts
    Q_arr = multiprod(params.Q, pts(ii).u(2:end).^2, 3, 2);
    m_forw = zeros(d,K); P_forw = zeros(d,d,K); m_forw(:,1) = zeros(d,1); P_forw(:,:,1) = params.prior_var_z;
    [m_forw(:,2:end),P_forw(:,:,2:end)] = kf_loop(zeros(d,1), params.prior_var_z, params.C, params.var_y, y(2:end), params.A, Q_arr);
    [pts(ii).m, pts(ii).P] = rts_smooth(m_forw, P_forw, params.A, cat(3,Q_arr,zeros(d)));
    
end

end

