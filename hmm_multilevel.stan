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
  // TODO
}

model {

  // Priors
  // TODO: PRIORS OVER POP LEVEL PARAMS, NOT SUBJ-LEVEL PARAMS
  target += gamma_lpdf(theta[1]-1 | 2, 2);
  target += gamma_lpdf(theta[2]-1 | 2, 2);
  target += beta_lpdf(phi[1,1] | 1.2, 1.2);
  target += beta_lpdf(phi[2,2] | 1.2, 1.2);
  
  // Multilevel model
  // TODO: SUBJECT-LEVEL PARAMS DRAWN FROM POP-LEVEL PARAMS
  for (s in 1:Ns) {
    phi[s,1,1] ~ this_subj_phi;
    phi[s,1,2] ~ 1-this_subj_phi; //don't think this line is needed? simplex is constrained so shouldn't contribute twice to log prob?
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
