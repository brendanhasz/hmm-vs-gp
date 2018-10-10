data {
    int<lower=1> N;      //number of datapoints
    real x[N];           //x values
    real<lower=0> rho;   //length scale
    real<lower=0> alpha; //marginal/output/signal standard deviation
    real<lower=0> sigma; //noise standard deviation
}

transformed data {
    matrix[N, N] K = cov_exp_quad(x, alpha, rho)
                     + diag_matrix(rep_vector(1e-10, N));
    matrix[N, N] L_K = cholesky_decompose(K);
}

parameters {}

model {}

generated quantities {
    // Declare variables
    vector[N] uf;
    vector[N] uy;
    vector[N] f;
    vector[N] y;

    // Function and samples in untransormed space
    uf = multi_normal_cholesky_rng(rep_vector(0, N), L_K);
    for (i in 1:N)
        uy[i] = normal_rng(uf[i], sigma);

    // Logit function and samples
    f = inv_logit(uf);
    y = inv_logit(uy);
}
