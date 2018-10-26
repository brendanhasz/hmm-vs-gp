data {
  int<lower=1> N;    //number of datapoints per trial
  int<lower=1> Nt;   //number of trials (total across all subjects)
  int<lower=1> Ns;   //number of subjects
  real x[N];         //x values (assume same for each trial/subject)
  int<lower=1,upper=Ns> S[Nt];//subject ID for each trial
  row_vector<lower=0, upper=1>[N] y[Nt]; //y values
}

transformed data {
  vector[N] mu;      //mean function
  real sum_ln_scale; //sum of scales for logit normal dists
  row_vector[N] logit_y[Nt]; //logit(y)
  mu = rep_vector(0, N);
  sum_ln_scale = 0;
  for (i in 1:Nt) //pre-compute contribution of logit normal scales
    sum_ln_scale += -sum(log(y[i])+log(1-y[i])); //to the posterior
  for (i in 1:Nt) //pre-compute logit(y) for each trial
    logit_y[i] = logit(y[i]);
}

parameters {
  // Per-subject parameters
  real<lower=0> rho[Ns];   //length scale
  real<lower=0> alpha[Ns]; //marginal/output/signal standard deviation
  real<lower=0> sigma[Ns]; //noise standard deviation
  
  // Population-level parameters
  real<lower=0> rho_a;   //alpha param of rho population gamma dist
  real<lower=0> rho_b;   //beta param of rho population gamma dist
  real<lower=0> alpha_a; //alpha param of rho population gamma dist
  real<lower=0> alpha_b; //beta param of rho population gamma dist
  real<lower=0> sigma_a; //alpha param of rho population gamma dist
  real<lower=0> sigma_b; //beta param of rho population gamma dist
}

transformed parameters {
  real<lower=0> rho_mu = rho_a/rho_b;       //pop mean length scale
  real<lower=0> alpha_mu = alpha_a/alpha_b; //pop mean signal std dev
  real<lower=0> sigma_mu = sigma_a/sigma_b; //pop mean noise std dev
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
  target += inv_gamma_lpdf(rho_mu | 2, 0.5);
  target += normal_lpdf(alpha_mu | 0, 2) + log(2); //half-normal dists
  target += normal_lpdf(sigma_mu | 0, 1) + log(2); //mult density by 2
  
  // Subject-level parameters drawn from pop-level distributions
  target += gamma_lpdf(rho | rho_a, rho_b);
  target += gamma_lpdf(alpha | alpha_a, alpha_b);
  target += gamma_lpdf(sigma | sigma_a, sigma_b);
  
  // Accumulate evidence over trials
  for (i in 1:Nt)
    target += multi_normal_lpdf(logit_y[i] | mu, K[S[i]]);
    
  // Add logit-normal scale terms to log posterior
  target += sum_ln_scale;
}
