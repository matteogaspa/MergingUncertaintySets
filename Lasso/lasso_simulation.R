rm(list = ls())
library(conformalInference)
library(dplyr)
library(ggplot2)
source("majorityVote.R")

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

#fix the u values
u1 <- 1/3
u2 <- 2/3

# Vote each point x0
out  <- lapply(conf.pred.ints, function(x) c(x$lo <= y0 & y0 <= x$up))
out  <- do.call(cbind, out)
rowMeans(out) > 1/2
cis <- matrix(NA, nrow = k, ncol = 2)
for(i in 1:k){
  cis[i,] <- c(conf.pred.ints[[i]]$lo, conf.pred.ints[[i]]$up)
}

cis_mv <- majority_vote(cis, rep(1/k, k), rho = 0.5)
cis_rmv1 <- majority_vote(cis, rep(1/k, k), rho = 0.5+u1/2)
cis_rmv2 <- majority_vote(cis, rep(1/k, k), rho = 0.5+u2/2)
cis_rv1 <- majority_vote(cis, rep(1/k, k), rho = u1)
cis_rv2 <- majority_vote(cis, rep(1/k, k), rho = u2)

all_cis <- rbind(cis, cis_mv, cis_rmv1, cis_rmv2,cis_rv1, cis_rv2) 
all_cis <- cbind(all_cis, c(1:23, 23, 24:25))
colnames(all_cis) <- c("low", "up", "n")
all_cis <- as.data.frame(all_cis)

my_col <- c(rep("Pred. Intervals", k), "Maj. Vote", rep("Rand. Maj. Vote", 3), rep("Rand. Vote", 2))

ggplot(all_cis, x=n) + 
  geom_segment(aes(x = n, xend = n, y = low, yend = up, color = my_col)) +
  scale_color_manual(values = c("Pred. Intervals" = "black", "Maj. Vote" = "blue", "Rand. Maj. Vote" = "red", "Rand. Vote" = "forestgreen")) +
  labs(title = "", x = "Tuning parameter", y = "Prediction Intervals", color = NULL) +
  theme_minimal() +
  theme(legend.position = "bottom")
                 


# Simulation -----
B   <- 10000             # number of replications
res <- matrix(NA, B, 3)  # results
len <- matrix(NA, B, 3)  # disj. ints
set.seed(123)
for(i in 1:B){
  # design matrix and outcome
  X <- matrix(rnorm(n*p), nrow = n, ncol = p)
  y <- X %*% beta + rnorm(n)
  
  # design matrix and outcome (test)
  X0 <- matrix(rnorm(n0*p), nrow = n0, ncol = p)
  y0 <- X0 %*% beta + rnorm(n0)
  
  conf.pred.ints <- lapply(funs, function(z) conformal.pred.split(X, y, X0, alpha = alpha/2,
                                                                  train.fun = z$train, predict.fun = z$predict, seed = i))
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
  if(i %% 50 == 0) cat("Iter: ", i, "\n")
}

colMeans(res); colMeans(len>1)
