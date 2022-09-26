function val = indicator(u, constraint)

if strcmpi(constraint.type,'none')
    val = 0;
elseif strcmpi(constraint.type,'a')
    if sum(sum(exp(-imag(u))>constraint.absorption.max+eps)) == 0 && sum(sum(exp(-imag(u))<constraint.absorption.min-eps)) == 0
        val = 0;
    else
        val = inf;
    end
else
    error("Invalid constraint. Should be 'A'(absorption) or 'none'.")
end

end


