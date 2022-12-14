function x = prox(v,gamma,n_iters,constraint)
global tau;
[n1,n2] = size(v);
w = zeros(n1,n2,2);
w_prev = zeros(n1,n2,2);
z = zeros(n1,n2,2);

for t = 1:n_iters
    w = z + 1/8/gamma*D(proj(v-gamma*DT(z),constraint));
    w = min(abs(w),tau).*exp(1i*angle(w));
    z = w + t/(t+3)*(w-w_prev);
    w_prev = w;
end

x = proj(v - gamma*DT(w),constraint);

end

