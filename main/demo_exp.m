% ========================================================================
% Introduction
% ========================================================================
% This code provides a simple demonstration of the projected refractive
% index framework for multi-wavelength phase retrieval.
% 
% Reference:
%   [1] Yunhui Gao and Liangcai Cao, "Projected refractive index framework 
%       for multi-wavelength phase retrieval," Optics Letters 47, 
%       5965-5968 (2022).
% -------------------------------------------------------------------------
% Author: Yunhui Gao (gyh21@mails.tsinghua.edu.cn)
% =========================================================================
%%
% =========================================================================
% Data processing
% ========================================================================= 
clear;clc;
close all;

% load function and data
addpath(genpath('./utils'))
addpath(genpath('../src'))
group_num = 1;
load(['../data/experiment/E',num2str(group_num),'/data.mat'])

K  = size(y,3);     % number of diversity measurements

% select area of interest
figure
[temp,rect] = imcrop(y(:,:,1));close;
if rem(size(temp,1),2) == 1
    rect(4) = rect(4) - 1;
end
if rem(size(temp,2),2) == 1
    rect(3) = rect(3) - 1;
end

y_crop = zeros(round(rect(4)),round(rect(3)),K);
for k = 1:K
    y_crop(:,:,k) = imcrop(y(:,:,k),rect);
end
y = y_crop;
N1 = size(y,1);
N2 = size(y,2);

% zero-pad the object to avoid convolution artifacts
kernelsize = params.dist*max(params.wavlen)/params.pxsize/2;    % diffraction kernel size
nullpixels = ceil(kernelsize/params.pxsize);                    % number of padding pixels

% forward model operators
Q  = @(x,k) propagate(x, params.dist,params.pxsize,params.wavlen(k),params.method); % forward propagation
QH = @(x,k) propagate(x,-params.dist,params.pxsize,params.wavlen(k),params.method); % Hermitian of Q: backward propagation
C  = @(x) imgcrop(x,nullpixels);        % image cropping operation
CT = @(x) zeropad(x,nullpixels);        % transpose of C: zero-padding operation
A  = @(x,k) C(Q(x,k));                  % overall sampling operation
AH = @(x,k) QH(CT(x),k);                % Hermitian of A

%%
% =========================================================================
% Compressive phase retrieval algorithm
% =========================================================================

clear functions     % release memory (if using puma)

% define the constraint
constraint.type = 'a';              % 'none': no constraint, 'a': absorption constraint only, 
constraint.absorption.max = 1.0;    % define the upper and lower bounds for the amplitude
constraint.absorption.min = 0;

% region for computing the errors
region.x1 = nullpixels+1;
region.x2 = nullpixels+N1;
region.y1 = nullpixels+1;
region.y2 = nullpixels+N2;

% initial guess
pha = zeros(N1+2*nullpixels,N2+2*nullpixels);
amp = ones(N1+2*nullpixels,N2+2*nullpixels);
u_init = pha + 1i*(-1)*log(amp);       

% iterations
n_iters    = 200;       % number of iterations (main loop)
n_subiters = 5;         % number of iterations (denoising)
gam = 2;                % step size (see the paper for details)

% regularization parameter tuning
alph = 10;
lam1 = 8e-3;
lam2 = 4e-3;

% options
opts.verbose = true;                                            % display status during the iterations
opts.errfunc = [];
opts.tau = @(iter) reg_param(iter, n_iters, alph, lam1, lam2);
opts.eta = 2;               % parameters for the incremental algorithm
opts.threshold = 1e-4;

% plot the regularization parameter
iter = 1:n_iters;
figure,plot(iter, opts.tau(iter));

% building blocks
myF     = @(u) F(u,y,A,K,params);
mydF    = @(u) dF(u,y,A,AH,K,params);
mydFk   = @(u,k) dFk(u,y,A,AH,k,params);
myR     = @(u) CCTV(u,constraint);                     % regularization function
myproxR = @(u,gam) prox(u,gam,n_subiters,constraint);  % proximal operator for the regularization function

% run the algorithm
[u_ref,~,~,~] = IPG(u_init,myF,mydFk,myR,myproxR,gam,10,K,opts);
[u_ref,J_ref,E_ref,runtimes_ref] = APG(u_ref,myF,mydF,myR,myproxR,gam,n_iters,opts);

%%
% =========================================================================
% Display results
% =========================================================================

% crop image to match the size of the sensor
u_ref_crop = u_ref(nullpixels+1:nullpixels+N1,nullpixels+1:nullpixels+N2);

% visualize the reconstructed image
figure
subplot(1,2,1),imshow(exp(-imag(u_ref_crop)),[]);colorbar
title('Retrieved amplitude','interpreter','latex','fontsize',14)
subplot(1,2,2),imshow(real(u_ref_crop),[]);colorbar
title('Retrieved phase','interpreter','latex','fontsize',14)
set(gcf,'unit','normalized','position',[0.2,0.3,0.6,0.4])

%%
% =========================================================================
% Implementation of the conventional algorithm (transmission model)
% =========================================================================
clear functions;

% algorithm settings
n_iters_trans = 100;         % number of iterations

% initialization
pha = zeros(N1,N2);           % initial phase
amp = ones(N1,N2);            % initial amplitude
x_trans = amp.*exp(1i*pha);   % initial object transmission function

% store runtimes
runtimes_trans = NaN(n_iters_trans,1);

% iteration
timer = tic;
for iter = 1:n_iters_trans
    if rem(iter,20) == 0; clear functions; end;
    for k = K:-1:1
        % convert phase to the next wavelength
        if k == K; k_prev = 1; else; k_prev = k + 1; end
        pha = params.wavlen(k_prev)/params.wavlen(k)*puma_ho(angle(x_trans),1);  % phase unwrapping
        % GS iterates
        x_pad = zeropad(abs(x_trans).*exp(1i*pha),nullpixels);  % zero-padding (support constraint)
        z = Q(x_pad,k);                                         % forward propagation
        z(nullpixels+1:end-nullpixels,nullpixels+1:end-nullpixels) = sqrt(y(:,:,k)).*exp(1i*angle(imgcrop(z,nullpixels)));    % intensity constraint
        x_trans = imgcrop(QH(z,k),nullpixels);                  % backward propagation
        x_trans = min(1,abs(x_trans)).*exp(1i*angle(x_trans));  % absorption constraint
    end
    % calculate runtime
    runtimes_trans(iter) = toc(timer);
    fprintf('iter: %4d | runtime: %5.1f s\n',iter, runtimes_trans(iter));
end

%%
% =========================================================================
% Display results
% =========================================================================

% visualize the reconstructed image
figure
subplot(1,2,1),imshow(abs(x_trans),[]);colorbar
title('Retrieved amplitude','interpreter','latex','fontsize',14)
subplot(1,2,2),imshow(angle(x_trans),[]);colorbar
title('Retrieved phase','interpreter','latex','fontsize',14)
set(gcf,'unit','normalized','position',[0.2,0.3,0.6,0.4])

%%
% =========================================================================
% Auxiliary functions
% =========================================================================

function v = F(u,y,A,K,params)
% =========================================================================
% Data-fidelity function.
% -------------------------------------------------------------------------
% Input:    - u      : The refractive object function.
%           - y      : Intensity image.
%           - A      : The sampling operator.
%           - K      : Number of diversity measurements.
%           - params : Physical parameters.
% Output:   - v      : Value of the fidelity function.
% =========================================================================
v = 0;
for k = 1:K
    x = exp(1i*params.wavlen(1)/params.wavlen(k)*real(u) - imag(u));
    v = v + 1/2/K * norm2(abs(A(x,k)) - sqrt(y(:,:,k)))^2;
end

function n = norm2(x)   % calculate the l2 vector norm
n = norm(x(:),2);
end

end


function g = dF(u,y,A,AH,K,params)
% =========================================================================
% Gradient of the data-fidelity function.
% -------------------------------------------------------------------------
% Input:    - u      : The refractive object function.
%           - y      : Intensity image.
%           - A      : The sampling operator.
%           - AH     : Hermitian of A.
%           - K      : Number of diversity measurements.
%           - params : Physical parameters.
% Output:   - g      : Wirtinger gradient.
% =========================================================================
g = zeros(size(u));

for k = 1:K
    
    g = g + 1/K * dFk(u,y,A,AH,k,params);
end

end


function g = dFk(u,y,A,AH,k,params)
% =========================================================================
% Gradient of the k-th term of the data-fidelity function.
% -------------------------------------------------------------------------
% Input:    - u      : The refractive object function.
%           - y      : Intensity image.
%           - A      : The sampling operator.
%           - AH     : Hermitian of A.
%           - K      : Number of diversity measurements.
%           - params : Physical parameters.
% Output:   - g      : Wirtinger gradient.
% =========================================================================
x = exp(1i*params.wavlen(1)/params.wavlen(k)*real(u) - imag(u));
v = A(x,k);
v = (abs(v) - sqrt(y(:,:,k))) .* exp(1i*angle(v));
v = conj(x).* AH(v,k);
g = (-1i/4) *((params.wavlen(1)/params.wavlen(k) + 1)* v + (-params.wavlen(1)/params.wavlen(k) + 1)*conj(v));
end


function u = imgcrop(x,cropsize)
% =========================================================================
% Crop the central part of the image.
% -------------------------------------------------------------------------
% Input:    - x        : Original image.
%           - cropsize : Cropping pixel number along each dimension.
% Output:   - u        : Cropped image.
% =========================================================================
u = x(cropsize+1:end-cropsize,cropsize+1:end-cropsize);
end


function u = zeropad(x,padsize)
% =========================================================================
% Zero-pad the image.
% -------------------------------------------------------------------------
% Input:    - x        : Original image.
%           - padsize  : Padding pixel number along each dimension.
% Output:   - u        : Zero-padded image.
% =========================================================================
u = padarray(x,[padsize,padsize],0);
end


function u = x2u(x)
% =========================================================================
% Convert the refractive object function into the object transmission
% function.
% -------------------------------------------------------------------------
% Input:    - x        : The object transmission function.
% Output:   - u        : The refractive object function.
% =========================================================================
pha = puma_ho(angle(x),1);
amp = abs(x);
u = pha + 1i*(-log(amp));
end


function x = u2x(u)
% =========================================================================
% Convert the object transmission function into the refractive object
% function.
% -------------------------------------------------------------------------
% Input:    - u        : The refractive object function.
% Output:   - x        : The object transmission function.
% =========================================================================
x = exp(1i*u);
end


function tau_val = reg_param(iter,n_iters,alph,tau1,tau2)
% =========================================================================
% Setting the regularization parameter tau during the iterations using an
% sigmoid function.
% -------------------------------------------------------------------------
% Input:    - iter     : The current iteration number.
%           - n_iters  : Total iteration numbers.
%           - alph     : Sigmoid function parameter.
%           - tau1     : Initial regularization parameter.
%           - tau2     : Final regularization parameter.
% Output:   - lam_val  : The current regularization parameter.
% =========================================================================
global tau;
tau = (tau1-tau2)*1./(1+exp(alph*(iter/n_iters - 1/2))) + tau2;
tau_val = tau;
end