data {
    int<lower=1> N;                 //number of datapoints
    real x[N];                      //x values
    vector<lower=0, upper=1>[N] y;  //y values (one N-dim datapoint)
}

transformed data {
    vector[N] mu = rep_vector(0, N);       //mean function
    real ln_scale = -sum(log(y)+log(1-y)); //scale for logit normal dists
}

parameters {
    real<lower=0> rho;   //length scale
    real<lower=0> alpha; //marginal/output/signal standard deviation
    real<lower=0> sigma; //noise standard deviation
}

model {
  
    // TODO: wait - why do we have to cholesky_decompose? 
    // we don't need to do that if we're not generating the mean line, right?
    
    // Decomposed covariance matrix
    matrix[N, N] K = cov_exp_quad(x, alpha, rho)
                     + diag_matrix(rep_vector(square(sigma), N));
    ///matrix[N, N] L_K = cholesky_decompose(K);

    // Priors
    target += inv_gamma_lpdf(rho | 2, 0.5);
    target += normal_lpdf(alpha | 0, 2) + log(2); //half-normal dists
    target += normal_lpdf(sigma | 0, 1) + log(2); //mult density by 2

    // Likelihood
    //target += multi_normal_cholesky_lpdf(logit(y) | mu, L_K);
    target += multi_normal_lpdf(logit(y) | mu, K);

    // Add scale such that likelihood integrates to 1 over y
    target += ln_scale;
    
    // TODO: handle multiple trials
    // TODO: multilevel model to handle multiple subjects
}
