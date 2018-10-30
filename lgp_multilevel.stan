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
  int<lower=1> Nt; //number of trials (total across all subjects)
  int<lower=1> Ns; //number of subjects
  vector[N] x;     //ind var (assume same across trials/subjects)
  int<lower=1,upper=Ns> S[Nt];//subject ID for each trial
  row_vector<lower=0, upper=1>[N] y[Nt]; //dependent variable
}

transformed data {
  vector[N] mu;      //mean function (vector of 0s)
  real sum_ln_scale; //sum of scales for logit normal dists
  row_vector[N] logit_y[Nt]; //logit(y)
  mu = rep_vector(0, N);
  sum_ln_scale = 0;
  for (i in 1:Nt) //pre-compute contribution of logit normal scales
    sum_ln_scale += -sum(log(y[i])+log(1-y[i]));
  for (i in 1:Nt) //pre-compute logit(y) for each trial
    logit_y[i] = logit(y[i]);
}

parameters {
  // Per-subject parameters
  vector<lower=0>[Ns] rho;   //length scale
  vector<lower=0>[Ns] alpha; //marginal/output/signal standard deviation
  vector<lower=0>[Ns] sigma; //noise standard deviation
  
  // Population-level parameters
  //real<lower=0> rho_m;   //mean of rho population dist
  //real<lower=0> rho_s;   //std of rho population dist
  //real<lower=0> alpha_m; //mean of alpha population dist
  //real<lower=0> alpha_s; //std param of alpha populationdist
  //real<lower=0> sigma_m; //mean param of sigma population dist
  //real<lower=0> sigma_s; //std param of sigma population dist
  
  // Population-level parameters
  real<lower=0> rho_a;   //alpha param of rho population gamma dist
  real<lower=0> rho_b;   //beta param of rho population gamma dist
  real<lower=0> alpha_a; //alpha param of alpha population gamma dist
  real<lower=0> alpha_b; //beta param of alpha population gamma dist
  real<lower=0> sigma_a; //alpha param of sigma population gamma dist
  real<lower=0> sigma_b; //beta param of sigma population gamma dist
}

transformed parameters {
  //real<lower=0> rho_mu = rho_a/rho_b;       //pop mean length scale
  real<lower=0> alpha_mu = alpha_a/alpha_b; //pop mean signal std dev
  real<lower=0> sigma_mu = sigma_a/sigma_b; //pop mean noise std dev
  
  
  // Population-level parameters
  //real<lower=0> rho_a;   //alpha param of rho population gamma dist
  //real<lower=0> rho_b;   //beta param of rho population gamma dist
  //real<lower=0> alpha_a; //alpha param of alpha population gamma dist
  //real<lower=0> alpha_b; //beta param of alpha population gamma dist
  //real<lower=0> sigma_a; //alpha param of sigma population gamma dist
  //real<lower=0> sigma_b; //beta param of sigma population gamma dist
}

model {
  // Covariance matrix for each subject (assume x same for each trial)
  // NOTE: this could be optimized, each off-diag will be identical w/ 
  // identically spaced x vals. So you really only need to do evaluate
  // the kernel function N/2 times, not N^2/2 times.
  matrix[N, N] K[Ns];
  for (s in 1:Ns) {
    K[s] = cov_exp_quad(x, alpha[s], rho[s]) + 
           diag_matrix(rep_vector(square(sigma[s]), N));
  }

  // Priors (on population-level params)
  //target += inv_gamma_lpdf(rho_mu | 2, 0.5);
  target += normal_lpdf(alpha_mu | 0, 2) + log(2); //half-normal dists
  target += normal_lpdf(sigma_mu | 0, 1) + log(2); //mult density by 2
  //TODO: also put priors on variance of dists: gamma dists?
  
  // Subject-level parameters drawn from pop-level distributions
  //target += gamma_lpdf(rho | rho_a, rho_b);
  target += inv_gamma_lpdf(rho | 2, 0.5); //TODO: DEBUGGER: is it just the rhos being drawn from a dist which is too wide causing the divergent transitions?
  target += gamma_lpdf(alpha | alpha_a, alpha_b);
  target += gamma_lpdf(sigma | sigma_a, sigma_b);
  // TODO: use non-centered parameterization for subj level alphas, sigmas, and rho (so might have to do log-normal population dists?)
  
  // Accumulate evidence over trials
  for (i in 1:Nt)
    target += multi_normal_lpdf(logit_y[i] | mu, K[S[i]]);
    
  // Add logit-normal scale terms to log posterior
  target += sum_ln_scale;
}
