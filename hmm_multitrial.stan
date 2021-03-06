data {
  int<lower=1> N; //number of observations per trial
  int<lower=1> Nt; //number of trials
  real<lower=0, upper=1> y[Nt, N]; //observations

}

parameters {
  simplex[2] phi[2];      //transition probabilities
  real<lower=1> theta[2]; //observation distribution params
}

model {

  // Priors
  target += beta_lpdf(phi[1,1] | 1.2, 1.2);
  target += beta_lpdf(phi[2,2] | 1.2, 1.2);
  target += gamma_lpdf(theta[1]-1 | 2, 2);
  target += gamma_lpdf(theta[2]-1 | 2, 2);

  // Compute the marginal probability over possible sequences
  {
    real acc[2];
    real gamma[N, 2];
    for (i in 1:Nt) { // accumulate evidence over trials
      gamma[1,1] = beta_lpdf(y[i,1] | 1, theta[1]);
      gamma[1,2] = beta_lpdf(y[i,2] | theta[2], 1);
      for (t in 2:N) {
        for (k in 1:2) {
          acc[1] = gamma[t-1, 1] + log(phi[1,k]);
          acc[2] = gamma[t-1, 2] + log(phi[2,k]);
          gamma[t,k] = log_sum_exp(acc);
        }
        gamma[t,1] += beta_lpdf(y[i,t] | 1, theta[1]);
        gamma[t,2] += beta_lpdf(y[i,t] | theta[2], 1);
      }
      target += log_sum_exp(gamma[N]);
    }
  }

}
