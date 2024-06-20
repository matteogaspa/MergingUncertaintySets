rm(list=ls())
library(ggplot2)
source("utils.R")

# Functions ----
nppr_algorithm <- function(x_t, r_t=1, g_t=1){
  x_t_ceil <- ceiling((x_t*g_t)/g_t)
  x_t_floor <- floor((x_t*g_t)/g_t)
  if(x_t_ceil==x_t_floor){
    y_t <- x_t
  }else{
    y_t <- sample(c(x_t_ceil, x_t_floor), 1, prob = c(g_t*(x_t-x_t_floor), g_t*(x_t_ceil-x_t)))
  }
  u_t <- sample(seq(0,g_t,by=1)/g_t, 1)
  z_t <- sample(c(y_t, u_t), 1, prob=c(r_t, 1-r_t))
  return(z_t)
}

nppr_algorithm <- Vectorize(nppr_algorithm, "x_t")

nppr_adj_mean <- function(z_t, r_t){
  z_t_adj <- z_t - 0.5*(1-r_t)
  return(mean(z_t_adj)/mean(r_t))
}

nppr_ci <- function(z_t, r_t, alpha = 0.05){
  mu_star <- nppr_adj_mean(z_t, r_t)
  n   <- length(z_t)
  a_t <- alpha/2
  q_t <- sqrt(log(1/a_t)/(2*n*mean(r_t)^2))
  return(c(mu_star-q_t, mu_star+q_t))
}

alpha_lev <- function(z_t, r_t, test_p){
  n <- length(z_t)
  theta_hat <- nppr_adj_mean(z_t, r_t)
  return(exp(-2*n*r_t^2*(test_p - theta_hat)^2))
}


## Confidence distr. -----
set.seed(1234)
n <- 100
k <- 10
test_point <- seq(0.01, 0.99, by = 0.005)
r_t <- (exp(2)-1)/(exp(2)+1)


# sample the common obs
samp <- runif(n*k/2)
# obtain the K different samples
samp_mat <- matrix(NA, nrow = n, ncol = k)
for(i in 1:k){
  rnd_obs <- sample(1:(n*k/2), n, replace = F)
  samp_mat[,i] <- samp[rnd_obs] 
}

mat_p <- matrix(NA, nrow = k, ncol = length(test_point))

for(i in 1:k){
  for(j in 1:length(test_point)){
    mat_p[i, j] <- alpha_lev(samp_mat[,i], r_t, test_point[j])
  }
}

alpha_levs <- 2*apply(mat_p, 2, function(x) quantile(x, 0.5))
alpha_median <- apply(mat_p, 2, function(x) quantile(x, 0.5))
u <- 0.1
alpha_median2 <- 2*apply(mat_p, 2, function(x) quantile(x, u))
u <- 0.2
alpha_median3 <- 2*apply(mat_p, 2, function(x) quantile(x, u))
u <- 0.3
alpha_median4 <- 2*apply(mat_p, 2, function(x) quantile(x, u))
u <- 0.4
alpha_median5 <- 2*apply(mat_p, 2, function(x) quantile(x, u))


cis_alpha <- matrix(NA, k, 2)
for(i in 1:k){
  cis_alpha[i,] <- nppr_ci(samp_mat[,i], rep(r_t, n), 0.1)
}

ic1 <- majority_vote(cis_alpha, rep(1/k, k))


# plot
df_plot2 <- cbind(test_point, t(mat_p), alpha_levs, alpha_median, alpha_median2, alpha_median3, alpha_median4, alpha_median5)
colnames(df_plot2) <- c("test_points", "cd1", "cd2", "cd3", "cd4", "cd5",
                        "cd6", "cd7", "cd8", "cd9", "cd10", "median2", "median", "rnd_median", "rnd_median1", "rnd_median2", "rnd_median3")
df_plot2 <- as.data.frame(df_plot2)

ggplot(df_plot2, aes(x = test_point)) +
  geom_line(aes(y=cd1), color = "gray", linetype = "solid", size = 0.5) + 
  geom_line(aes(y=cd2), color = "gray", linetype = "solid", size = 0.5) + 
  geom_line(aes(y=cd3), color = "gray", linetype = "solid", size = 0.5) + 
  geom_line(aes(y=cd4), color = "gray", linetype = "solid", size = 0.5) + 
  geom_line(aes(y=cd5), color = "gray", linetype = "solid", size = 0.5) + 
  geom_line(aes(y=cd6), color = "gray", linetype = "solid", size = 0.5) + 
  geom_line(aes(y=cd7), color = "gray", linetype = "solid", size = 0.5) + 
  geom_line(aes(y=cd8), color = "gray", linetype = "solid", size = 0.5) + 
  geom_line(aes(y=cd9), color = "gray", linetype = "solid", size = 0.5) + 
  geom_line(aes(y=cd10), color = "gray", linetype = "solid", size = 0.5) + 
  geom_line(aes(y = rnd_median), color = "forestgreen", linetype = "solid", size = 0.5) +
  geom_line(aes(y = rnd_median1), color = "forestgreen", linetype = "solid", size = 0.5) +
  geom_line(aes(y = rnd_median2), color = "forestgreen", linetype = "solid", size = 0.5) +
  geom_line(aes(y = rnd_median3), color = "forestgreen", linetype = "solid", size = 0.5) +
  geom_line(aes(y = median), color = "red", linetype = "solid", size = 1) +
  geom_line(aes(y = median2), color = "blue", linetype = "solid", size = 1) +
  
  labs(title = "",
       x = "",
       y = "Confidence distribution") +
  theme_minimal()

