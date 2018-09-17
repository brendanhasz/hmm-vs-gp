# Parameters
N = 1000
x = seq(0, 10, l=N)
rho = 1
alpha = 2
sigma = 0.3

# Simulate
sim_data = list(N=N, x=x, rho=rho, alpha=alpha, sigma=sigma)
sim_fit = stan(file='simulate_lgp.stan', data=sim_data, iter=1, 
               chains=1, seed=123456, algorithm="Fixed_param")

# Extract simulated output
f = extract(sim_fit)$f[1,]
y = extract(sim_fit)$y[1,]

# Plot
plot(x, f, type="l", lwd=2)
points(x, y, col="black", pch=16, cex=0.4)
