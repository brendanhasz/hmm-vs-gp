// Generates data from a 2-state Hidden Markov Model 
// with Beta-distributed observation probabilities

data {
  int<lower=1> N;         //number of observations to generate
  vector[2] phi[2];       //transition probabilities
  real<lower=0> theta[2]; //observation distribution params
}

transformed data { 
  real thetas[2,2]; //just so we don't need if/elses in generated quantities...
  thetas[1,1] = 1;
  thetas[1,2] = theta[1];
  thetas[2,1] = theta[2];
  thetas[2,2] = 1;
}

parameters {}

model {}

generated quantities {

  // Declare variables
  real y[N];         //observations
  int<lower=1> s[N]; //hidden states

  // Generate hidden state sequence and observations
  s[1] = categorical_rng(rep_vector(0.5, 2)); //start at random state
  y[1] = beta_rng(thetas[s[1],1], thetas[s[1],2]); //generate observation
  for (t in 2:N) {
    s[t] = categorical_rng(phi[s[t-1]]); //new state
    y[t] = beta_rng(thetas[s[t],1], thetas[s[t],2]); //generate observation
  }

}
