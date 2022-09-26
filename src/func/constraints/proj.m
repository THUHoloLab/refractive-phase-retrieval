function y = proj(u,constraint)

if strcmpi(constraint.type,'none')
    y = u;
elseif strcmpi(constraint.type,'a')
    y = real(u) + 1i*min(max(imag(u), -log(constraint.absorption.max)), -log(constraint.absorption.min));
else
    error("Invalid constraint. Should be 'A'(absorption) or 'none'.")
end

