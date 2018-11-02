data {
    int<lower=1> N; //number of datapoints
    real x[N];      //x values
    vector[N] y;    //y values (one N-dimensional datapoint)
}

transformed data {
    vector[N] mu = rep_vector(0, N);
}

parameters {
    real<lower=0> rho;
    real<lower=0> alpha;
    real<lower=0> sigma;
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
    y ~ multi_normal_cholesky(mu, L_K);
}
