data {
  int<lower=1> N; //number of datapoints per trial
  int<lower=1> M; //number of trials
  real x[N];      //x values (assume same for each trial)
  row_vector<lower=0, upper=1>[N] y[M]; //y values
}

transformed data {
  vector[N] mu;      //mean function
  real ln_scale[M];  //scale for logit normal dists
  mu = rep_vector(0, N);
  for (i in 1:M) 
    ln_scale[i] = -sum(log(y[i])+log(1-y[i])); 
}

parameters {
  real<lower=0> rho;   //length scale
  real<lower=0> alpha; //marginal/output/signal standard deviation
  real<lower=0> sigma; //noise standard deviation
}

model {
  // Covariance matrix (assume x same for each trial)
  // NOTE: this could be optimized, each off-diag will be identical w/ 
  // identically spaced x vals. So you really only need to do evaluate
  // the kernel function N/2 times, not N^2/2 times.
  matrix[N, N] K = cov_exp_quad(x, alpha, rho) + 
                   diag_matrix(rep_vector(square(sigma), N));

  // Priors
  target += inv_gamma_lpdf(rho | 2, 0.5);
  target += normal_lpdf(alpha | 0, 2) + log(2); //half-normal dists
  target += normal_lpdf(sigma | 0, 1) + log(2); //mult density by 2
  
  // Accumulate evidence over trials
  for (i in 1:M)
    target += multi_normal_lpdf(logit(y[i]) | mu, K);
    
  // Add scales such that likelihood integrates to 1 over y
  target += sum(ln_scale);
  
  // TODO: multilevel model to handle multiple subjects
}
