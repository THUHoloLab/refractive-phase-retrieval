function [x,J_vals,E_vals,runtimes] = IPG(x_init,F,dFk,R,proxR,gam,n_iters,K,opts)
% =========================================================================
% Proxiimal gradient algorithm with incremental updates.
% -------------------------------------------------------------------------
% Input:    - x_init   : Initial guess.
%           - F        : Fidelity function.
%           - dFk      : Gradient of the fidelity function w.r.t. the k-th measurement.
%           - R        : Regularization function.
%           - proxR    : Proximity operator of the regularization function.
%           - gam      : Initial step size.
%           - n_iters  : Number of iterations.
%           - K        : Number of measurements.
%           - opts     : Other options.
% Output:   - x        : Final estimate.
%           - J_vals   : Objective function values.
%           - E_vals   : Error metrics.
%           - runtimes : Runtimes.
% =========================================================================

% initialization
x = x_init;

% cache data
J_vals = NaN(n_iters+1,1);  % objective function values
E_vals = NaN(n_iters+1,1);  % error metrics
runtimes = NaN(n_iters,1);  % runtimes

if isa(opts.tau,'function_handle')
    opts.tau(0);
end

J_vals(1) = F(x) + R(x);
if isa(opts.errfunc,'function_handle')
    E_vals(1) = opts.errfunc(x);
end

if opts.display
    figure
    set(gcf,'unit','normalized','position',[0.25,0.3,0.5,0.4],'color','w')
end

% set timer
timer = tic;

% main loop
for iter = 1:n_iters
    
    if isa(opts.tau,'function_handle')
        opts.tau(iter);
    end
    
    % incremental gradient update
    for k = 1:K
        x = x - gam*dFk(x,k);
    end
    
    % proximal update
    x = proxR(x,gam);
    
    J_vals(iter+1) = F(x) + R(x);
    
    % record runtime
    runtimes(iter) = toc(timer);
    
    % calculate error metric
    if isa(opts.errfunc,'function_handle')
        E_vals(iter+1) = opts.errfunc(x);
    end
    
    % display status
    if opts.verbose
        fprintf('iter: %4d | objective: %10.4e | stepsize: %2.2e | runtime: %5.1f s\n', ...
                iter, J_vals(iter+1), gam, runtimes(iter));
    end
    
    % adaptive step size selection
    if (J_vals(iter) - J_vals(iter+1)) / J_vals(iter) < opts.threshold
        gam = gam / opts.eta;
    end

    % display results
    if opts.display
        subplot(1,2,1),imshow(exp(-imag(x)),[]);colorbar
        title('Retrieved amplitude','fontsize',12)
        subplot(1,2,2),imshow(real(x),[]);colorbar
        title('Retrieved phase','fontsize',12)
        drawnow;
    end
    
end

if opts.display
    close
end

end

