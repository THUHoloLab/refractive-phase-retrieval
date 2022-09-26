function re = relative_error_2d(u_es,u_gt,region)

u_es = u_es(region.x1:region.x2,region.y1:region.y2);
u_gt = u_gt(region.x1:region.x2,region.y1:region.y2);

pha_es = real(u_es);
pha_gt = real(u_gt);

u_es = u_es - mean(pha_es(:)) + mean(pha_gt(:));

re = norm2(u_es - u_gt) / norm2(u_gt);

function val = norm2(u)
    val = norm(u(:),2);
end

end

