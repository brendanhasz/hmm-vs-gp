functions {
  /**
   * Return a covariance matrix using the exponentiated quadratic 
   * kernel, optimized for when elements of x are linearly spaced.
   * 
   * @param x Vector of linearly-spaced independent variable values
   * @param a Signal standard deviation (or amplitude) parameter
   * @param r Length scale parameter
   * @param s Noise standard deviation parameter
   * @param N Number of observations in x
   *
   * @return Covariance matrix (N-by-N)
   */
  matrix cov_exp_quad_lin(real[] x, real a, real r, real s, int N) {
    matrix[N, N] K;
    real a2;
    real r2;
    real s2;
    real a2s2;
    real dx;
    real dx2;
    real val;
    int k;
    a2 = a*a;
    r2 = -0.5/(r*r);
    s2 = s*s;
    a2s2 = a2+s2;
    for (i in 1:N) //fill in diagonal
      K[i,i] = a2s2;
    for (i in 2:N) {
      dx = x[i]-x[1]; //difference
      dx2 = dx*dx;    //squared difference
      val = a2*exp(r2*dx2); //value for this off-diagonal
      for (j in 1:(N-i+1)) {
        k = i+j-1;
        K[j,k] = val; //off-diagonals
        K[k,j] = val; //are identical
      }
    }
    return(K);
  }
}

data {
  int<lower=1> N;  //number of datapoints per trial
  int<lower=1> Nt; //number of trials
  real x[N];       //x values (assume same for each trial)
  row_vector<lower=0, upper=1>[N] y[Nt]; //y values
}

transformed data {
  vector[N] mu;      //mean function
  real ln_scale[Nt]; //scale for logit normal dists
  mu = rep_vector(0, N);
  for (i in 1:Nt) 
    ln_scale[i] = -sum(log(y[i])+log(1-y[i])); 
}

parameters {
  real<lower=0> rho;   //length scale
  real<lower=0> alpha; //marginal/output/signal standard deviation
  real<lower=0> sigma; //noise standard deviation
}

model {
  // Covariance matrix (assume x same for each trial)
  matrix[N, N] K = cov_exp_quad(x, alpha, rho) + 
                   diag_matrix(rep_vector(square(sigma), N));
  //matrix[N, N] K = cov_exp_quad_lin(x, alpha, rho, sigma, N); 
  
  // Priors
  target += inv_gamma_lpdf(rho | 2, 0.5);
  target += normal_lpdf(alpha | 0, 2) + log(2); //half-normal dists
  target += normal_lpdf(sigma | 0, 1) + log(2); //mult density by 2
  
  // Accumulate evidence over trials
  for (i in 1:Nt)
    target += multi_normal_lpdf(logit(y[i]) | mu, K);
    
  // Add scales such that likelihood integrates to 1 over y
  target += sum(ln_scale);
  
}
