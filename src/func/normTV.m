function n = normTV(x)

global tau;
g = D(x);
n = tau * norm1(g(:,:,1)) + tau * norm1(g(:,:,2));



function v = norm1(x)
    v = norm(x(:),1);
end

end
