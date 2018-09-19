
setwd("~/Code/hmm-vs-gp")

# Packages
library(rstan)
library(ggplot2)
library(bayesplot)
rstan_options(auto_write = TRUE)
options(mc.cores = parallel::detectCores())
seed = 723456

# Colors
c_light <- c("#DCBCBC")
c_mid <- c("#B97C7C")
c_dark <- c("#8F2727")
color_scheme_set("red")

# Data
N = 100
x = seq(0, 5, l=N)

# Parameters
rho = 0.5
alpha = 2
sigma = 0.3

# Simulate
sim_data = list(N=N, x=x, rho=rho, alpha=alpha, sigma=sigma)
sim_fit = stan(file='simulate_lgp.stan', data=sim_data, iter=1, 
               chains=1, seed=seed, algorithm="Fixed_param")

# Extract simulated output
f = extract(sim_fit)$f[1,]
y = extract(sim_fit)$y[1,]

# Plot
plot(x, f, type="l", lwd=2, ylim=c(0,1))
points(x, y, col=c_mid, pch=16, cex=0.5)
title(main='Simulated Gaussian Process')

# Fit
stan_rdump(c("N", "x", "y"), file="simulated_lgp.data.R")
fit_data = read_rdump("simulated_lgp.data.R")
fit = stan(file='lgp.stan', data=fit_data, seed=seed)

# Launch shinystan
#library(shinystan)
#launch_shinystan(fit)

# Show results of fit
print(fit)

# Check no chains hit max tree depth
check_treedepth(fit)

# Check the energy Bayesian Fraction of Missing Information
check_energy(fit)

# Check divergences
check_divergences(fit)

# Plot posteriors
posterior = as.array(fit)
mcmc_dens(posterior, pars=c("rho", "alpha", "sigma"))

# Plot trace per chain
mcmc_trace(posterior, pars=c("rho", "alpha", "sigma"))

# Plot density per chain
mcmc_dens_overlay(posterior, pars=c("rho", "alpha", "sigma"))

# Plot true vs posterior for rho
posterior = extract(fit)
par(mfrow=c(1, 3))

hist(posterior$rho, main="", xlab="rho", ylab="",
     col=c_dark, yaxt='n', breaks=20)
abline(v=rho, col=c_light, lwd=3)

hist(posterior$alpha, main="", xlab="alpha", ylab="",
     col=c_dark, yaxt='n', breaks=20)
abline(v=alpha, col=c_light, lwd=3)

hist(posterior$sigma, main="", xlab="sigma", ylab="",
     col=c_dark, yaxt='n', breaks=20)
abline(v=sigma, col=c_light, lwd=3)


