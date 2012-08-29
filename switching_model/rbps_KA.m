function [ pts ] = rbps_KA( params, filt_pts_array, filt_wts_array, y )
%RBPS_FULL Run a RB particle smoother using Kim's approximation

d = params.d;
K = params.K;
Nf = params.Nf;
Ns = params.Ns;

z_var_arr = zeros(1, 1, params.K);
y_var_arr = zeros(1, 1, params.K);
A_arr = zeros(params.d,params.d,params.K);

% Initialise particle array
pts = struct('u', repmat({zeros(1,K)}, Ns, 1), 'm', repmat({zeros(d,K)}, Ns, 1), 'P', repmat({zeros(d,d,K)}, Ns, 1));

% Loop through particles
for ii = 1:Ns
    
    fprintf(1, '*** Smoothing particle %u.\n', ii);
    
    % Initialise particle by sampling from final filtering frame
    ind = randsample(Nf, 1, true, exp(filt_wts_array{K}));
    pts(ii).u(K) = filt_pts_array{K}(ind).u(K);
    
    if pts(ii).u(K) == 0
        z_var_arr(:,:,K) = params.var_z_0;
        y_var_arr(:,:,K) = params.var_y_0;
        A_arr(:,:,K) = params.A0;
    elseif pts(ii).u(K) == 1
        z_var_arr(:,:,K) = params.var_z_1;
        y_var_arr(:,:,K) = params.var_y_1;
        A_arr(:,:,K) = params.A1;
    end
    
    % Loop backwards through time
    for kk = K-1:-1:2
        
        % Initialise array of sampling weights
        samp_wts = zeros(Nf,1);
        
        % Loop through filter particles
        for jj = 1:Nf
            
            % Nonlinear bit
            if pts(ii).u(kk+1) == filt_pts_array{kk}(jj).u
                nonlin_prob = log(1-params.p_switch);
            else
                nonlin_prob = log(params.p_switch);
            end;
            
            % Calculate sampling weight
            samp_wts(jj) = filt_wts_array{kk}(jj) + nonlin_prob;
            
        end
        
        % Normalise weights
        samp_wts = samp_wts - logsumexp(samp_wts);
        
        % Sample
        ind = randsample(Nf, 1, true, exp(samp_wts));
        pts(ii).u(kk) = filt_pts_array{kk}(ind).u;
        
        if pts(ii).u(kk) == 0
            z_var_arr(:,:,kk) = params.var_z_0;
            y_var_arr(:,:,kk) = params.var_y_0;
            A_arr(:,:,kk) = params.A0;
        elseif pts(ii).u(kk) == 1
            z_var_arr(:,:,kk) = params.var_z_1;
            y_var_arr(:,:,kk) = params.var_y_1;
            A_arr(:,:,kk) = params.A1;
        end
        
        
    end
    
    Q_arr = multiprod(z_var_arr, params.Q);
    R_arr = multiprod(y_var_arr, params.R);   
    
    % Run a RTS smoother to ascertain the linear parts
    m_forw = zeros(d,K); P_forw = zeros(d,d,K); m_forw(:,1) = zeros(d,1); P_forw(:,:,1) = params.prior_var_z;
    [m_forw(:,2:end),P_forw(:,:,2:end)] = kf_loop(zeros(d,1), params.prior_var_z, params.C, R_arr, y(:,2:end), A_arr(:,:,2:end), Q_arr(:,:,2:end));
    [pts(ii).m, pts(ii).P] = rts_smooth(m_forw, P_forw, A_arr(:,:,2:end), Q_arr(:,:,2:end));
    
end

end

