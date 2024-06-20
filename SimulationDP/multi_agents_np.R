rm(list=ls())
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


# Simulations -----
B <- 5000
n <- 10
cov <- rep(NA,B)
cis <- matrix(NA, nrow = B, ncol = 2)
set.seed(1234)
for(i in 1:B){
  samp <- runif(n)
  samp_nppr <- nppr_algorithm(samp)
  cis[i,] <- nppr_ci(samp_nppr, rep(.5, n), 0.05)
  cov[i] <- I(cis[i,1] <= 0.5 & 0.5 <= cis[i,2])
}
mean(cov)

# Simulations with multiple agents -----
# Common (pubblic) dataset
B <- 10000
n <- 100
k <-10
alpha <- 0.1
r_t <- (exp(2)-1)/(exp(2)+1)

# 90% obs shared
ns <- rep(10, 10)
cov90 <- l90 <- matrix(NA, nrow = B, ncol = 12)
for(i in 1:B){
  samp <- runif(n)
  M_int <- matrix(NA, k, 2)
  for(j in 1:length(ns)){
    samp_plus <- c(samp, runif(ns[j]))
    cis <- nppr_ci(nppr_algorithm(samp_plus), rep(r_t, n+ns[j]), alpha)
    M_int[j,] <- cis
    cov90[i,j] <- I(cis[1] < 0.5 & 0.5 < cis[2])
    l90[i,j] <- cis[2]-cis[1]
  }
  cov90[i, 11] <- I(mean(cov90[i,1:10])>0.5)
  cis <- majority_vote(M_int, rep(1,k))
  l90[i,11] <- cis[2]-cis[1]
  us <- runif(1, 0.5, 1)
  cov90[i, 12] <- I(mean(cov90[i,1:10])>us)
  cis <- majority_vote(M_int, rep(1,k), us)
  l90[i,12] <- cis[2]-cis[1]
  if(i %% 100 == 0) cat("iter:", i, "\n")
}
colMeans(cov90)
colMeans(l90)

# 75% obs shared
ns <- rep(50, 10)
cov75 <- l75 <- matrix(NA, nrow = B, ncol = 12)
for(i in 1:B){
  samp <- runif(n)
  M_int <- matrix(NA, k, 2)
  for(j in 1:length(ns)){
    samp_plus <- c(samp, runif(ns[j]))
    cis <- nppr_ci(nppr_algorithm(samp_plus), rep(r_t, n+ns[j]), alpha)
    M_int[j,] <- cis
    cov75[i,j] <- I(cis[1] < 0.5 & 0.5 < cis[2])
    l75[i,j] <- cis[2]-cis[1]
  }
  cov75[i, 11] <- I(mean(cov75[i,1:10])>0.5)
  cis <- majority_vote(M_int, rep(1,k))
  l75[i,11] <- cis[2]-cis[1]
  us <- runif(1, 0.5, 1)
  cov75[i, 12] <- I(mean(cov75[i,1:10])>us)
  cis <- majority_vote(M_int, rep(1,k), us)
  l75[i,12] <- cis[2]-cis[1]
  if(i %% 100 == 0) cat("iter:", i, "\n")
}
colMeans(cov75)
colMeans(l75)

# 50% obs shared
ns <- rep(100, 10)
cov50 <- l50 <- matrix(NA, nrow = B, ncol = 12)
for(i in 1:B){
  samp <- runif(n)
  M_int <- matrix(NA, k, 2)
  for(j in 1:length(ns)){
    samp_plus <- c(samp, runif(ns[j]))
    cis <- nppr_ci(nppr_algorithm(samp_plus), rep(r_t, n+ns[j]), alpha)
    M_int[j,] <- cis
    cov50[i,j] <- I(cis[1] < 0.5 & 0.5 < cis[2])
    l50[i,j] <- cis[2]-cis[1]
  }
  cov50[i, 11] <- I(mean(cov50[i,1:10])>0.5)
  cis <- majority_vote(M_int, rep(1,k))
  l50[i,11] <- cis[2]-cis[1]
  us <- runif(1, 0.5, 1)
  cov50[i, 12] <- I(mean(cov50[i,1:10])>us)
  cis <- majority_vote(M_int, rep(1,k), us)
  l50[i,12] <- cis[2]-cis[1]
  if(i %% 100 == 0) cat("iter:", i, "\n")
}
colMeans(cov50)
colMeans(l50)

cov_sh <- list(cov90=cov90,cov75=cov75,cov50=cov50,
               l90=l90,l75=l75,l50=l50)
save(cov_sh, file = "cov_sh.RData")


# Simulations with multiple agents 2 -----
B <- 10000
n <- 100
k <- 10
alpha <- 0.1
r_t <- (exp(2)-1)/(exp(2)+1)
set.seed(1234)

# Agent k share 50% of the data with agent k-1
cov50 <- l50 <- matrix(0, nrow = B, ncol = 15)
for(i in 1:B){
  samp <- runif(n)
  cis <- nppr_ci(nppr_algorithm(samp), rep(r_t, n), alpha)
  cov50[i,k] <- I(cis[1] < 0.5 & 0.5 < cis[2])
  l50[i,k] <- cis[2]-cis[1]
  M_int <- matrix(NA, k, 2)
  M_int[k,] <- cis
  for(j in 1:(k-1)){
    rnd_obs <- sample(1:n, n/2, replace = F)
    samp_plus <- c(samp[rnd_obs], runif(n/2))
    cis <- nppr_ci(nppr_algorithm(samp_plus), rep(r_t, n), alpha)
    M_int[j,] <- cis
    cov50[i,j] <- I(cis[1] < 0.5 & 0.5 < cis[2])
    l50[i,j] <- cis[2]-cis[1]
  }
  cov50[i, 11] <- I(mean(cov50[i,1:k])>0.5)
  cis <- majority_vote(M_int, rep(1,k))
  l50[i,11] <- cis[2]-cis[1]
  us <- runif(1, 0.5, 1)
  cov50[i, 12] <- I(mean(cov50[i,1:10])>us)
  cis <- majority_vote(M_int, rep(1,k), us)
  l50[i,12] <- cis[2]-cis[1]  
  us2 <- runif(1, 0, 1)
  cov50[i, 13] <- I(mean(cov50[i,1:10])>us2)
  cis <- majority_vote(M_int, rep(1,k), us2)
  l50[i,13] <- cis[2]-cis[1] 
  cis <- exch_majority_vote(M_int)
  if(!is.na(cis[1])){
    for(l in 1:nrow(cis)){
      cov50[i,14] <- cov50[i,14] + I(cis[l,1] <= 0.5 & 0.5 <= cis[l,2])
      l50[i,14] <- l50[i,14] + (cis[l,2] - cis[l,1])
    }
  }
  cis <- exch_rand_majority_vote(M_int)
  if(!is.na(cis[1])){
    for(l in 1:nrow(cis)){
      cov50[i,15] <- cov50[i,15] + I(cis[l,1] <= 0.5 & 0.5 <= cis[l,2])
      l50[i,15] <- l50[i,15] + (cis[l,2] - cis[l,1])
    }
  }
  if(i %% 100 == 0) cat("iter:", i, "\n")
}
colMeans(cov50); colMeans(l50,na.rm=T)

# Agent k share 75% of the data with agent k-1
cov75 <- l75 <- matrix(0, nrow = B, ncol = 15)
set.seed(1234)
for(i in 1:B){
  samp <- runif(n)
  cis <- nppr_ci(nppr_algorithm(samp), rep(r_t, n), alpha)
  cov75[i,k] <- I(cis[1] < 0.5 & 0.5 < cis[2])
  l75[i,k] <- cis[2]-cis[1]
  M_int <- matrix(NA, k, 2)
  M_int[k,] <- cis
  for(j in 1:(k-1)){
    rnd_obs <- sample(1:n, n/2, replace = F)
    samp_plus <- c(samp[rnd_obs], runif(n/2))
    cis <- nppr_ci(nppr_algorithm(samp_plus), rep(r_t, n), alpha)
    M_int[j,] <- cis
    cov75[i,j] <- I(cis[1] < 0.5 & 0.5 < cis[2])
    l75[i,j] <- cis[2]-cis[1]
  }
  cov75[i, 11] <- I(mean(cov75[i,1:k])>0.5)
  cis <- majority_vote(M_int, rep(1,k))
  l75[i,11] <- cis[2]-cis[1]
  us <- runif(1, 0.5, 1)
  cov75[i, 12] <- I(mean(cov75[i,1:10])>us)
  cis <- majority_vote(M_int, rep(1,k), us)
  l75[i,12] <- cis[2]-cis[1]  
  us2 <- runif(1, 0, 1)
  cov75[i, 13] <- I(mean(cov75[i,1:10])>us2)
  cis <- majority_vote(M_int, rep(1,k), us2)
  l75[i,13] <- cis[2]-cis[1]  
  if(!is.na(cis[1])){
    for(l in 1:nrow(cis)){
      cov75[i,14] <- cov75[i,14] + I(cis[l,1] <= 0.5 & 0.5 <= cis[l,2])
      l75[i,14] <- l75[i,14] + (cis[l,2] - cis[l,1])
    }
  }
  cis <- exch_rand_majority_vote(M_int)
  if(!is.na(cis[1])){
    for(l in 1:nrow(cis)){
      cov75[i,15] <- cov75[i,15] + I(cis[l,1] <= 0.5 & 0.5 <= cis[l,2])
      l75[i,15] <- l75[i,15] + (cis[l,2] - cis[l,1])
    }
  }
  if(i %% 100 == 0) cat("iter:", i, "\n")
}
colMeans(cov75)

# Agent k share 90% of the data with agent k-1
cov90 <- l90 <- matrix(0, nrow = B, ncol = 15)
set.seed(1234)
for(i in 1:B){
  samp <- runif(n)
  cis <- nppr_ci(nppr_algorithm(samp), rep(r_t, n), alpha)
  cov90[i,k] <- I(cis[1] < 0.5 & 0.5 < cis[2])
  l90[i,k] <- cis[2]-cis[1]
  M_int <- matrix(NA, k, 2)
  M_int[k,] <- cis
  for(j in 1:(k-1)){
    rnd_obs <- sample(1:n, n/2, replace = F)
    samp_plus <- c(samp[rnd_obs], runif(n/2))
    cis <- nppr_ci(nppr_algorithm(samp_plus), rep(r_t, n), alpha)
    M_int[j,] <- cis
    cov90[i,j] <- I(cis[1] < 0.5 & 0.5 < cis[2])
    l90[i,j] <- cis[2]-cis[1]
  }
  cov90[i, 11] <- I(mean(cov90[i,1:k])>0.5)
  cis <- majority_vote(M_int, rep(1,k))
  l90[i,11] <- cis[2]-cis[1]
  us <- runif(1, 0.5, 1)
  cov90[i, 12] <- I(mean(cov90[i,1:10])>us)
  cis <- majority_vote(M_int, rep(1,k), us)
  l90[i,12] <- cis[2]-cis[1]  
  us2 <- runif(1, 0, 1)
  cov90[i, 13] <- I(mean(cov90[i,1:10])>us2)
  cis <- majority_vote(M_int, rep(1,k), us2)
  l90[i,13] <- cis[2]-cis[1] 
  if(!is.na(cis[1])){
    for(l in 1:nrow(cis)){
      cov90[i,14] <- cov90[i,14] + I(cis[l,1] <= 0.5 & 0.5 <= cis[l,2])
      l90[i,14] <- l90[i,14] + (cis[l,2] - cis[l,1])
    }
  }
  cis <- exch_rand_majority_vote(M_int)
  if(!is.na(cis[1])){
    for(l in 1:nrow(cis)){
      cov90[i,15] <- cov90[i,15] + I(cis[l,1] <= 0.5 & 0.5 <= cis[l,2])
      l90[i,15] <- l90[i,15] + (cis[l,2] - cis[l,1])
    }
  }
  if(i %% 100 == 0) cat("iter:", i, "\n")
}
colMeans(cov90)
cov_ma <- list(cov90=cov90,cov75=cov75,cov50=cov50,
               l90=l90, l75=l75, l50=l50)
save(cov_ma, file = "cov_ma.RData")

## Draw from a common set -----
B <- 10000
n <- 100
k <- 10
alpha <- 0.1
r_t <- (exp(2)-1)/(exp(2)+1)
cov <- lng <- matrix(0, nrow = B, ncol = 15)
set.seed(1234)
for(i in 1:B){
  samp <- runif(n*k/2)
  M_int <- matrix(NA, k, 2)
  for(j in 1:k){
    rnd_obs <- sample(1:(n*k/2), n, replace = F)
    samp_plus <- c(samp[rnd_obs])
    cis <- nppr_ci(nppr_algorithm(samp_plus), rep(r_t, n), alpha)
    M_int[j,] <- cis
    cov[i,j] <- I(cis[1] < 0.5 & 0.5 < cis[2])
    lng[i,j] <- cis[2]-cis[1]
  }
  cov[i, 11] <- I(mean(cov[i,1:k])>0.5)
  cis <- majority_vote(M_int, rep(1,k))
  lng[i,11] <- cis[2]-cis[1]
  us <- runif(1, 0.5, 1)
  cov[i, 12] <- I(mean(cov[i,1:10])>us)
  cis <- majority_vote(M_int, rep(1,k), us)
  lng[i,12] <- cis[2]-cis[1]
  us2 <- runif(1, 0, 1)
  cov[i, 13] <- I(mean(cov[i,1:10])>us2)
  cis <- majority_vote(M_int, rep(1,k), us2)
  lng[i,13] <- cis[2]-cis[1]
  if(!is.na(cis[1])){
    for(l in 1:nrow(cis)){
      cov[i,14] <- cov[i,14] + I(cis[l,1] <= 0.5 & 0.5 <= cis[l,2])
      lng[i,14] <- lng[i,14] + (cis[l,2] - cis[l,1])
    }
  }
  cis <- exch_rand_majority_vote(M_int)
  if(!is.na(cis[1])){
    for(l in 1:nrow(cis)){
      cov[i,15] <- cov[i,15] + I(cis[l,1] <= 0.5 & 0.5 <= cis[l,2])
      lng[i,15] <- lng[i,15] + (cis[l,2] - cis[l,1])
    }
  }
  if(i %% 100 == 0) cat("iter:", i, "\n")
}
colMeans(cov); colMeans(lng, na.rm=T)

# results -----
library(xtable)
library(dplyr)

# agent
results <- matrix(NA, 8, 6)

results[1,1] <- lng[1,1]
results[2,1] <- colMeans(cov)[1:k]%>%mean

results[3,1] <- cov_ma$l90[1,1]
results[4,1] <- colMeans(cov90)[1:k]%>%mean

results[5,1] <- cov_ma$l75[1,1]
results[6,1] <- colMeans(cov75)[1:k]%>%mean

results[7,1] <- cov_ma$l50[1,1]
results[8,1] <- colMeans(cov50)[1:k]%>%mean

# majority vote
results[1,2] <- colMeans(lng)[k+1]
results[2,2] <- colMeans(cov)[k+1]

results[3,2] <- colMeans(cov_ma$l90)[k+1]
results[4,2] <- colMeans(cov90)[k+1]

results[5,2] <- colMeans(cov_ma$l75)[k+1]
results[6,2] <- colMeans(cov75)[k+1]

results[7,2] <- colMeans(cov_ma$l50)[k+1]
results[8,2] <- colMeans(cov50)[k+1]

# rnd majority vote
results[1,3] <- colMeans(lng, na.rm = T)[k+2]
results[2,3] <- colMeans(cov)[k+2]

results[3,3] <- colMeans(cov_ma$l90, na.rm = T)[k+2]
results[4,3] <- colMeans(cov90)[k+2]

results[5,3] <- colMeans(cov_ma$l75, na.rm = T)[k+2]
results[6,3] <- colMeans(cov75)[k+2]

results[7,3] <- colMeans(cov_ma$l50, na.rm=T)[k+2]
results[8,3] <- colMeans(cov50)[k+2]

# randomized union
results[1,4] <- colMeans(lng, na.rm = T)[k+3]
results[2,4] <- colMeans(cov)[k+3]

results[3,4] <- colMeans(cov_ma$l90, na.rm = T)[k+3]
results[4,4] <- colMeans(cov90)[k+3]

results[5,4] <- colMeans(cov_ma$l75, na.rm = T)[k+3]
results[6,4] <- colMeans(cov75)[k+3]

results[7,4] <- colMeans(cov_ma$l50, na.rm=T)[k+3]
results[8,4] <- colMeans(cov50)[k+3]

# exchangeable union
results[1,5] <- colMeans(lng, na.rm = T)[k+4]
results[2,5] <- colMeans(cov)[k+4]

results[3,5] <- colMeans(cov_ma$l90, na.rm = T)[k+4]
results[4,5] <- colMeans(cov90)[k+4]

results[5,5] <- colMeans(cov_ma$l75, na.rm = T)[k+4]
results[6,5] <- colMeans(cov75)[k+4]

results[7,5] <- colMeans(cov_ma$l50, na.rm=T)[k+4]
results[8,5] <- colMeans(cov50)[k+4]

# randomized exchangeable union
results[1,6] <- colMeans(lng, na.rm = T)[k+5]
results[2,6] <- colMeans(cov)[k+5]

results[3,6] <- colMeans(cov_ma$l90, na.rm = T)[k+5]
results[4,6] <- colMeans(cov90)[k+5]

results[5,6] <- colMeans(cov_ma$l75, na.rm = T)[k+5]
results[6,6] <- colMeans(cov75)[k+5]

results[7,6] <- colMeans(cov_ma$l50, na.rm=T)[k+5]
results[8,6] <- colMeans(cov50)[k+5]

colnames(results) <- c("Ck", "CM", "CR", "CU", "CE", "C*")
results <- as.data.frame(results)

print(xtable(results, digits = 4), include.rownames=F)



