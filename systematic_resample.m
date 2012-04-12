function [ Nchild, parent ] = systematic_resample( weights, Np_out )
%SYSTEMATIC_RESAMPLE Calculates number of children for each particle
%according to a systematic resampling scheme

% weights are assumed to be linear

% Number of particles in
Np_in = length(weights);

% Generate random index array
u = (1/Np_out)*ones(Np_out, 1);
u(1) = rand/Np_out;
u = cumsum(u);

% Generate cumulative weight array
w_sum = cumsum(weights);

% Enumerate offspring
parent = zeros(Np_out,1);
Nchild = zeros(Np_in,1);
Nchild(1) = sum(u < w_sum(1));
parent(1:Nchild(1)) = 1;
jj = Nchild(1);
for ii = 2:Np_in
    Nchild(ii) = sum((u < w_sum(ii))&(u > w_sum(ii-1)));
    parent(jj+1:jj+Nchild(ii)) = ii;
    jj = jj + Nchild(ii);
end

end

