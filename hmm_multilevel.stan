data {
  int<lower=1> N;     //number of observations per trial
  int<lower=1> Nt;    //number of trials
  int<lower=1> Ns;    //number of subjects
  int<lower=1> S[Nt]; //subject id
  real<lower=0, upper=1> y[Nt, N]; //observations
}

parameters {
  // Per-subject parameters (non-centered parameterization)
  real phi_tilde[Ns,2];   //transition probabilities
  real theta_tilde[Ns,2]; //observation distribution params
  
  // Population-level parameters
  real<lower=0,upper=1> phi_m[2]; //mean of phi population dists
  real<lower=0> phi_s[2];   //std of phi population dists
  real<lower=1> theta_m[2]; //mean of theta population dists
  real<lower=0> theta_s[2]; //std of theta population dists
}

transformed parameters {
  // Per-subject parameters
  simplex[2] phi[Ns,2];      //transition probabilities
  real<lower=1> theta[Ns,2]; //observation distribution params

  // Non-centered parameterization of per-subject parameters
  for (s in 1:Ns) {
    phi[s,1,1] = inv_logit(logit(phi_m[1])+phi_s[1]*phi_tilde[s,1]);
    phi[s,2,2] = inv_logit(logit(phi_m[2])+phi_s[2]*phi_tilde[s,2]);
    phi[s,1,2] = 1 - phi[s,1,1];
    phi[s,2,1] = 1 - phi[s,2,2];
    theta[s,1] = 1+exp(log(theta_m[1]-1)+theta_s[1]*theta_tilde[s,1]);
    theta[s,2] = 1+exp(log(theta_m[2]-1)+theta_s[2]*theta_tilde[s,2]);
  }
}

model {
  
  // Priors for each of the 2 states
  for (i in 1:2) {
    // Priors (on population-level params)
    target += beta_lpdf(phi_m[i] | 1.2, 1.2);
    target += gamma_lpdf(theta_m[i]-1 | 2, 2);
    target += normal_lpdf(phi_s[i] | 0, 1) + log(2);
    target += normal_lpdf(theta_s[i] | 0, 2) + log(2);
  
    // Subject-level parameters drawn from pop-level distributions
    // (non-centered parameterizations)
    target += normal_lpdf(phi_tilde[,i] | 0, 1);   //log(rho) ~ normal(rho_m, rho_s)
    target += normal_lpdf(theta_tilde[,i] | 0, 1); //log(rho) ~ normal(rho_m, rho_s)
  
    // Jacobian adjustments for GLM parts of model
    for (s in 1:Ns)
      target += -log(phi[s,i,i]*(1-phi[s,i,i]));
    target += -sum(log(theta[,i]));
  }

  // Compute the marginal probability over possible sequences
  {
    real acc[2];
    real gamma[N, 2];
    for (i in 1:Nt) { // accumulate evidence over trials
      gamma[1,1] = beta_lpdf(y[i,1] | 1, theta[S[i],1]);
      gamma[1,2] = beta_lpdf(y[i,2] | theta[S[i],2], 1);
      for (t in 2:N) {
        for (k in 1:2) {
          acc[1] = gamma[t-1, 1] + log(phi[S[i],1,k]);
          acc[2] = gamma[t-1, 2] + log(phi[S[i],2,k]);
          gamma[t,k] = log_sum_exp(acc);
        }
        gamma[t,1] += beta_lpdf(y[i,t] | 1, theta[S[i],1]);
        gamma[t,2] += beta_lpdf(y[i,t] | theta[S[i],2], 1);
      }
      target += log_sum_exp(gamma[N]);
    }
  }

}
