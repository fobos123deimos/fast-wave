function wavefunction = wavefunction_MATLAB_1(n, x, prec)
    
    digits(prec);
    n = vpa(n);
    x = vpa(x);

    norm = (2^(-0.5*n)) * (gamma(n+1)^(-0.5)) * (pi^(-0.25));

    H = hermiteH(n,x);

    wavefunction = norm * exp(-0.5 * x.^2) .* H;

end