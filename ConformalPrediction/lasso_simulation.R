rm(list = ls())
library(conformalInference)
library(dplyr)
library(ggplot2)
source("utils.R")
# simulation p > n
# lasso with different \lambda values
n  <- 100    # number of obs. in the train set
n0 <- 1      # number of obs. in the test set
p  <- 120    # number of predictors
m  <- 10     # number of active predictors

# \beta values
set.seed(123)
beta <- c(rnorm(m, mean = 0, sd = 2), rep(0, p-m))

# Example -----
# design matrix and outcome
X <- matrix(rnorm(n*p), nrow = n, ncol = p)
y <- X %*% beta + rnorm(n)

# design matrix and outcome (test)
X0 <- matrix(rnorm(n0*p), nrow = n0, ncol = p)
y0 <- X0 %*% beta + rnorm(n0)

# lasso for all parameters
lambda <- exp(seq(-4, 1.5, length = 20))
k      <- length(lambda)
funs   <- list()
for(i in 1:length(lambda)){
  funs[[i]] <- lasso.funs(lambda = lambda[i], standardize = F, intercept = F)
}


# Obtain a conformal prediction interval for each X0 with level (1-\alpha/2)
alpha <- 0.1
conf.pred.ints <-lapply(funs, function(z) conformal.pred.split(X, y, X0, alpha = alpha/2,
                                                               train.fun = z$train, predict.fun = z$predict, seed = 123))

us <- 1/2

# Vote each point x0
out  <- lapply(conf.pred.ints, function(x) c(x$lo <= y0 & y0 <= x$up))
out  <- do.call(cbind, out)
rowMeans(out) > 1/2
cis <- matrix(NA, nrow = k, ncol = 2)
for(i in 1:k){
  cis[i,] <- c(conf.pred.ints[[i]]$lo, conf.pred.ints[[i]]$up)
}

cis_mv    <- majority_vote(cis, rep(1/k, k), rho = 0.5)
cis_rmv   <- majority_vote(cis, rep(1/k, k), rho = 0.5+us/2)
cis_u     <- majority_vote(cis, rep(1/k, k), rho = us)
cis_exc   <- exch_majority_vote(cis)
#cis_exc_r <- exch_rand_majority_vote(cis)#run with u<-0.75 in the initial set

all_cis           <- rbind(cis, cis_mv, cis_rmv, cis_u, cis_exc)
all_cis           <- cbind(all_cis, c(1:22, 22, 23:24))
colnames(all_cis) <- c("low", "up", "n")
all_cis           <- as.data.frame(all_cis)
my_col            <- c(rep("a", k), "b", "c", "c", "d", "e")
all_cis$type      <- my_col
my_labs           <- c(expression(C[k]), expression(C^M), expression(C^R), expression(C^U), expression(C^pi))


# Plots -----
ggplot(all_cis, x = n, aes(color = my_col, shape = my_col)) + 
  geom_segment(aes(x = n, xend = n, y = low, yend = up, color = my_col)) +
  geom_point(aes(x = n, y = low, color = my_col)) +
  geom_point(aes(x = n, y = up, color = my_col)) +
  scale_color_manual(values = c("a"="black", "b"="red", "c"="blue", "d"="forestgreen", "e"="purple", "f"="salmon4"), labels = my_labs) +
  scale_shape_manual(name = "", values = c("a"=16, "b"=2, "c"=4, "d"=5, "e"=6, "f"=8), labels = my_labs) +
  labs(title = "", x = "Tuning parameter", y = "Prediction Intervals", color = NULL) +
  theme_minimal() +
  theme(legend.position = "bottom")


ggplot(all_cis, aes(x = n, color = type, shape = type)) + 
  geom_segment(aes(x = n, xend = n, y = low, yend = up, color = type)) +
  geom_point(aes(x = n, y = low, shape = type, color = type)) +
  geom_point(aes(x = n, y = up, shape = type, color = type)) +
  scale_color_manual(values = c("a"="black", "b"="red", "c"="blue", "d"="forestgreen", "e"="purple", "f"="salmon4"), labels = my_labs) +
  scale_shape_manual(values = c("a"=16, "b"=21, "c"=15, "d"=17, "e"=18), labels = my_labs) +
  labs(title = "", x = "Tuning parameter", y = "Prediction Intervals", color = NULL, shape = NULL) +
  theme_minimal() +
  theme(legend.position = "bottom")




# Simulation different weight -----
B   <- 10000            # number of replications
res <- matrix(0, B, 5)  # results
len <- matrix(0, B, 5)  # disj. ints
set.seed(123)
for(i in 1:B){
  # design matrix and outcome
  X <- matrix(rnorm(n*p), nrow = n, ncol = p)
  y <- X %*% beta + rnorm(n)
  
  # design matrix and outcome (test)
  X0 <- matrix(rnorm(n0*p), nrow = n0, ncol = p)
  y0 <- X0 %*% beta + rnorm(n0)
  
  conf.pred.ints <- lapply(funs, function(z) conformal.pred.split(X, y, X0, alpha = alpha/2,
                                                                  train.fun = z$train, predict.fun = z$predict))
  los <- lapply(conf.pred.ints, function(x) x$lo)
  los <- do.call(c, los)
  ups <- lapply(conf.pred.ints, function(x) x$up)
  ups <- do.call(c, ups)
  cis <- cbind(los, ups)
  out    <- lapply(conf.pred.ints, function(x) c(x$lo <= y0 & y0 <= x$up))
  out    <- do.call(cbind, out)
  vote   <- rowMeans(out)
  u1 <- runif(1, 0.5, 1)
  u2 <- runif(1)
  res[i,1] <- I(vote > 0.5)
  res[i,2] <- I(vote > u1)
  res[i,3] <- I(vote > u2)
  ci_m <- majority_vote(cis, rep(1, k), 0.5)
  ci_r <- majority_vote(cis, rep(1, k), u1)
  ci_u <- majority_vote(cis, rep(1, k), u2)
  len[i,1] <- ifelse(is.na(ci_m)[1], 0, nrow(ci_m))
  len[i,2] <- ifelse(is.na(ci_r)[1], 0, nrow(ci_r))
  len[i,3] <- ifelse(is.na(ci_u)[1], 0, nrow(ci_u))
  
  ci_e <- exch_majority_vote(cis)
  if(is.na(ci_e[1])){
    res[i,4] <- 0
  }else{
    for(j in 1:nrow(ci_e)){
      res[i,4] <- res[i,4] + as.numeric(I(ci_e[j,1] <= y0 & y0 <= ci_e[j,2]))
    }
  }
  len[i,4] <- ifelse(is.na(ci_e)[1], 0, nrow(ci_e))
  
  ci_er <- exch_rand_majority_vote(cis)
  if(is.na(ci_er[1])){
    res[i,5] <- 0
  }else{
    for(j in 1:nrow(ci_er)){
      res[i,5] <- res[i,5] + as.numeric(I(ci_er[j,1] <= y0 & y0 <= ci_er[j,2]))
    }
  }
  len[i,5] <- ifelse(is.na(ci_er)[1], 0, nrow(ci_er))
  
  if(i %% 50 == 0) cat("Iter: ", i, "\n")
}

colMeans(res);colMeans(len>1)


# Simulation different weights -----
B   <- 10000             # number of replications
res <- matrix(0, B, 5)  # results
len <- matrix(0, B, 5)  # size ints
set.seed(123)
for(i in 1:B){
  # design matrix and outcome
  X <- matrix(rnorm(n*p), nrow = n, ncol = p)
  y <- X %*% beta + rnorm(n)
  
  # design matrix and outcome (test)
  X0 <- matrix(rnorm(n0*p), nrow = n0, ncol = p)
  y0 <- X0 %*% beta + rnorm(n0)
  
  conf.pred.ints <- lapply(funs, function(z) conformal.pred.split(X, y, X0, alpha = alpha/2,
                                                                  train.fun = z$train, predict.fun = z$predict))
  los <- lapply(conf.pred.ints, function(x) x$lo)
  los <- do.call(c, los)
  ups <- lapply(conf.pred.ints, function(x) x$up)
  ups <- do.call(c, ups)
  cis <- cbind(los, ups)
  
  u1   <- runif(1, 0.5, 1)
  
  ci_0 <- majority_vote(cis, rep(1, k), u1)
  ci_1 <- majority_vote(cis, c(rep(3, k/2), rep(1, k/2)), u1)
  ci_2 <- majority_vote(cis, c(rep(1, k/2), rep(3, k/2)), u1)
  ci_3 <- majority_vote(cis, 1:20, u1)
  ci_4 <- majority_vote(cis, 20:1, u1)
  
  res[i,1] <- covr_fun(ci_0, y0)
  res[i,2] <- covr_fun(ci_1, y0)
  res[i,3] <- covr_fun(ci_2, y0)
  res[i,4] <- covr_fun(ci_3, y0)
  res[i,5] <- covr_fun(ci_4, y0)
  
  len[i,1] <- size_fun(ci_0)
  len[i,2] <- size_fun(ci_1)
  len[i,3] <- size_fun(ci_2)
  len[i,4] <- size_fun(ci_3)
  len[i,5] <- size_fun(ci_4)
  
  if(i %% 50 == 0) cat("Iter: ", i, "\n")
}

colMeans(res);colMeans(len)

# no rand
# Simulation different weights -----
B   <- 10000             # number of replications
res <- matrix(0, B, 5)  # results
len <- matrix(0, B, 5)  # size ints
res_ci <- cov_ci <- matrix(0, B, 20)

set.seed(123)
for(i in 1:B){
  # design matrix and outcome
  X <- matrix(rnorm(n*p), nrow = n, ncol = p)
  y <- X %*% beta + rnorm(n)
  
  # design matrix and outcome (test)
  X0 <- matrix(rnorm(n0*p), nrow = n0, ncol = p)
  y0 <- X0 %*% beta + rnorm(n0)
  
  conf.pred.ints <- lapply(funs, function(z) conformal.pred.split(X, y, X0, alpha = alpha/2,
                                                                  train.fun = z$train, predict.fun = z$predict))
  los <- lapply(conf.pred.ints, function(x) x$lo)
  los <- do.call(c, los)
  ups <- lapply(conf.pred.ints, function(x) x$up)
  ups <- do.call(c, ups)
  cis <- cbind(los, ups)
  res_ci[i,] <- ups-los
  cov_ci[i,] <- apply(cis, 1, function(x) covr_fun(x, y0))
  u1   <- .5
  
  ci_0 <- majority_vote(cis, rep(1, k), u1)
  ci_1 <- majority_vote(cis, c(rep(3, k/2), rep(1, k/2)), u1)
  ci_2 <- majority_vote(cis, c(rep(1, k/2), rep(3, k/2)), u1)
  ci_3 <- majority_vote(cis, 1:20, u1)
  ci_4 <- majority_vote(cis, 20:1, u1)
  
  res[i,1] <- covr_fun(ci_0, y0)
  res[i,2] <- covr_fun(ci_1, y0)
  res[i,3] <- covr_fun(ci_2, y0)
  res[i,4] <- covr_fun(ci_3, y0)
  res[i,5] <- covr_fun(ci_4, y0)
  
  len[i,1] <- size_fun(ci_0)
  len[i,2] <- size_fun(ci_1)
  len[i,3] <- size_fun(ci_2)
  len[i,4] <- size_fun(ci_3)
  len[i,5] <- size_fun(ci_4)
  
  if(i %% 50 == 0) cat("Iter: ", i, "\n")
}

colMeans(res);colMeans(len)
min(colMeans(res_ci)); max(colMeans(res_ci)); mean(colMeans(res_ci))
colMeans(cov_ci)[which.min(colMeans(res_ci))]; colMeans(cov_ci)[which.max(colMeans(res_ci))]; mean(colMeans(cov_ci))
