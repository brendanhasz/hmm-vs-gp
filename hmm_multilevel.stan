data {
  int<lower=1> N;  //number of observations per trial
  int<lower=1> Nt; //number of trials
  int<lower=1> Ns; //number of subjects
  int<lower=1> S[Nt]; //subject id
  real<lower=0, upper=1> y[Nt, N]; //observations
}

parameters {
  // Per-subject parameters
  simplex[2] phi[Ns,2];      //transition probabilities
  real<lower=1> theta[Ns,2]; //observation distribution params
  
  // Population-level parameters
  vector<lower=1>[2] phi_a;   //alpha param of phi's population beta dist
  vector<lower=1>[2] phi_b;   //beta param of phi's population beta dist
  vector<lower=0>[2] theta_a; //alpha param of theta's pop. gamma dist
  vector<lower=0>[2] theta_b; //beta param of theta's population gamma dist
}

transformed parameters {
  vector<lower=0,upper=1>[2] phi_mu = phi_a./(phi_a+phi_b); //pop mean recursive trans prob
  vector<lower=1>[2] theta_mu = theta_a./theta_b+1; //pop mean obs dist param
  //TODO: lambda param for beta dist (total count) w/ parieto prior? see pg 286 of stan manual
}

model {

  // Priors (on population-level params)
  target += beta_lpdf(phi_mu | 1.2, 1.2);
  target += gamma_lpdf(theta_mu-1 | 2, 2);
  
  // Subject-level parameters drawn from pop-level distributions
  target += beta_lpdf(phi[,1,1] | phi_a[1], phi_b[1]);
  target += beta_lpdf(phi[,2,2] | phi_a[2], phi_b[2]);
  target += gamma_lpdf(theta[,1]-1 | theta_a[1], theta_b[1]);
  target += gamma_lpdf(theta[,2]-1 | theta_a[2], theta_b[2]);
  
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
