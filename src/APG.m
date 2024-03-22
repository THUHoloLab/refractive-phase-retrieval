function [x,J_vals,E_vals,runtimes] = APG(x_init,F,dF,R,proxR,gamma,n_iters,opts)
% =========================================================================
% The accelerated proximal gradient (APG) algorithm.
% -------------------------------------------------------------------------
% Input:    - x_init   : Initial guess.
%           - F        : Fidelity function.
%           - dF       : Gradient of the fidelity function.
%           - R        : Regularization function.
%           - proxR    : Proximity operator of the regularization function.
%           - gam      : Initial step size.
%           - n_iters  : Number of iterations.
%           - opts     : Other options.
% Output:   - x        : Final estimate.
%           - J_vals   : Objective function values.
%           - E_vals   : Error metrics.
%           - runtimes : Runtimes.
% =========================================================================

% initialization
x = x_init;
z = x;
J_vals = NaN(n_iters+1,1);
E_vals = NaN(n_iters+1,1);
runtimes = NaN(n_iters,1);

if isa(opts.tau,'function_handle')
        opts.tau(0);
end
    
J_vals(1) = F(x)+R(x);
if isa(opts.errfunc,'function_handle')
    E_vals(1) = opts.errfunc(z);
end

if opts.display
    figure
    set(gcf,'unit','normalized','position',[0.25,0.3,0.5,0.4],'color','w')
end

iter_nes = 1;
timer = tic;
for iter = 1:n_iters
    
    if isa(opts.tau,'function_handle')
        opts.tau(iter);
    end
    
    % proximal gradient update
    x_next = proxR(z - gamma*dF(z),gamma);
    J_next = J(x_next);
    if J_next >= J_vals(iter)   % if larger: reduce to ista iteration
        iter_nes = 1;
        while true
            x_next = proxR(x - gamma*dF(x),gamma);
            J_next = J(x_next);
            if J_next <= J_vals(iter)
                break
            end
            gamma = gamma / 2; 
        end
    end
    J_vals(iter+1) = J_next;
    z = x_next + (iter_nes/(iter_nes+3))*(x_next - x);
    
    % record runtime
    runtimes(iter) = toc(timer);
    
    % calculate error metric
    if isa(opts.errfunc,'function_handle')
        E_vals(iter+1) = opts.errfunc(z);
    end
    
    % print status
    if opts.verbose
        fprintf('iter: %4d | objective: %10.4e | stepsize: %2.2e | runtime: %5.1f s\n', ...
                iter, J_vals(iter+1), gamma, runtimes(iter));
    end

    % display results
    if opts.display
        subplot(1,2,1),imshow(exp(-imag(x)),[]);colorbar
        title('Retrieved amplitude','fontsize',12)
        subplot(1,2,2),imshow(real(x),[]);colorbar
        title('Retrieved phase','fontsize',12)
        drawnow;
    end
    
    x = x_next;
    iter_nes = iter_nes + 1;
end

if opts.display
    close
end


function val = J(x)
val = F(x) + R(x);
end

end

