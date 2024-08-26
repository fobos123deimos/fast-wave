function wavefunction = wavefunction_MATLAB_3(n,x,prec)
    
    digits(prec);
    x = vpa(x);
    x_size = numel(x);

    wavefunction = vpa(zeros(x_size, n+1));

    for index = 1:n+1

        norm = (2^(-0.5*vpa(index-1))) * (gamma(vpa(index-1)+1)^(-0.5)) * (pi^(-0.25));
    
        H = hermiteH(vpa(index-1),x);
    
        wavefunction(:,index) = norm * exp(-0.5 * x^2) * H;
    end

end