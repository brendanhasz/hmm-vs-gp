data {
    int<lower=1> N;                 //number of datapoints
    real x[N];                      //x values
    vector<lower=0, upper=1>[N] y;  //y values (one N-dimensional datapoint)
}

transformed data {
    vector[N] mu = rep_vector(0, N);
}

parameters {
    real<lower=0> rho;   //length scale
    real<lower=0> alpha; //marginal/output/signal standard deviation
    real<lower=0> sigma; //noise standard deviation
}

model {
    // Decomposed covariance matrix
    matrix[N, N] K = cov_exp_quad(x, alpha, rho)
                     + diag_matrix(rep_vector(square(sigma), N));
    matrix[N, N] L_K = cholesky_decompose(K);

    // Priors
    rho ~ inv_gamma(5, 5);
    alpha ~ normal(0, 1);
    sigma ~ normal(0, 1);

    // Likelihood
    target += multi_normal_cholesky_lpdf(logit(y) | mu, L_K);

    // TODO: handle multiple trials
    // TODO: multilevel model to handle multiple subjects
}
