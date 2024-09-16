function wavefunction_result = wavefunction_MATLAB_2(n,x,prec)
  
  digits(prec);
  x = vpa(x);
  x_size = numel(x);

  wavefunction = vpa(zeros(n+1, x_size));

  wavefunction(1,:) = vpa((pi^(-0.25)) * exp(-(x.^2) / 2));
  wavefunction(2,:) = vpa((2*x .*wavefunction(1,:)) / sqrt(2));

  for index = 3:n+1
    wavefunction(index,:) = 2*x .*(wavefunction(index-1,:) / sqrt(2*(index-1))) - sqrt((index-2)/(index-1))*wavefunction(index-2,:);    
  end
  
  wavefunction_result = wavefunction(n+1,:);

end

