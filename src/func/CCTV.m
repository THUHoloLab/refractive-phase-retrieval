function val = CCTV(x,constraint)

val = normTV(x) + indicator(x,constraint);

end

