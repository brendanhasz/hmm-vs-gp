// 2-state Hidden Markov Model 
// with Beta-distributed observation probabilities

data {
  int<lower=1> N;              //number of observations
  real<lower=0, upper=1> y[N]; //observations
}

parameters {
  simplex[2] phi[2];      //transition probabilities
  real<lower=0> theta[2]; //observation distribution params
}

model {

  // Priors
  theta ~ 1 + gamma(2, 2);
  phi ~ beta(2, 2);

  // Compute the marginal probability over possible sequences
  {
    real acc[2];
    real gamma[N, 2];
    gamma[1,1] = beta_lpdf(y[1] | 1, theta[1]);
    gamma[1,2] = beta_lpdf(y[2] | theta[2], 1);
    for (t in 2:N) {
      for (k in 1:2) {
        acc[1] = gamma[t-1, 1] + log(phi[1,k]) + beta_lpdf(y[t] | 1, theta[1]);
        acc[2] = gamma[t-1, 2] + log(phi[2,k]) + beta_lpdf(y[t] | theta[2], 1);
        gamma[t,k] = log_sum_exp(acc);
      }
    }
    target += log_sum_exp(gamma[N]);
  }

  // TODO: handle multiple trials
  // TODO: multilevel model to handle multiple subjects

}