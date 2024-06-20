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
lambda <- 1
funs   <- lasso.funs(lambda = lambda, standardize = F, intercept = F)


# Obtain a conformal prediction interval for each X0 with level (1-\alpha/2)
alpha <- 0.1
B   <- 5000             
ks  <- c(1, 5, 10, 20, 30, 50)


res_1 <- vector("list", 5)  # results
len_1 <- vector("list", 5)  # results
nrw_1 <- vector("list", 5)  # results
for(i in 1:length(ks)){
  res_1[[i]] <- len_1[[i]] <- nrw_1[[i]] <- matrix(NA, B, 2)
}
res_3 <- res_2 <- res_1; len_3 <- len_2 <- len_1; nrw_3 <- nrw_2 <- nrw_1; 
tau1 <- 0.25
tau2 <- 0.50
tau3 <- 0.75


set.seed(1234)

for(l in 1:length(ks)){
  k <- ks[l]
  for(i in 1:B){
    # design matrix and outcome
    X <- matrix(rnorm(n*p), nrow = n, ncol = p)
    y <- X %*% beta + rnorm(n)
    
    # design matrix and outcome (test)
    X0 <- matrix(rnorm(n0*p), nrow = n0, ncol = p)
    y0 <- X0 %*% beta + rnorm(n0)
    
    data_list <- vector("list", k)
    for(j in 1:k){
      data_list[[j]]$X <- X
      data_list[[j]]$y <- y
      data_list[[j]]$X0 <- X0
    }
    
    conf.pred.ints <- lapply(data_list, function(z) conformal.pred.split(z$X, z$y, z$X0, alpha = alpha*(1-tau1),
                                                                         train.fun = funs$train, predict.fun = funs$predict))
    los <- lapply(conf.pred.ints, function(x) x$lo)
    los <- do.call(c, los)
    ups <- lapply(conf.pred.ints, function(x) x$up)
    ups <- do.call(c, ups)
    cis <- cbind(los, ups)
    c_m <- majority_vote(cis, rep(1/k, k), tau1)
    c_e <- exch_majority_vote(cis, tau1)
    
    res_1[[l]][i,1] <- sum(apply(c_m, 1, function(x) c(x[1] <= y0 & y0 <= x[2])))
    res_1[[l]][i,2] <- sum(apply(c_e, 1, function(x) c(x[1] <= y0 & y0 <= x[2])))
    len_1[[l]][i,1] <- sum(apply(c_m, 1, function(x) x[2]-x[1]))
    len_1[[l]][i,2] <- sum(apply(c_e, 1, function(x) x[2]-x[1]))
    nrw_1[[l]][i,1] <- nrow(c_m)
    nrw_1[[l]][i,2] <- nrow(c_e)
    
    conf.pred.ints <- lapply(data_list, function(z) conformal.pred.split(z$X, z$y, z$X0, alpha = alpha*(1-tau2),
                                                                         train.fun = funs$train, predict.fun = funs$predict))
    los <- lapply(conf.pred.ints, function(x) x$lo)
    los <- do.call(c, los)
    ups <- lapply(conf.pred.ints, function(x) x$up)
    ups <- do.call(c, ups)
    cis <- cbind(los, ups)
    c_m <- majority_vote(cis, rep(1/k, k), tau2)
    c_e <- exch_majority_vote(cis, tau2)
    
    res_2[[l]][i,1] <- sum(apply(c_m, 1, function(x) c(x[1] <= y0 & y0 <= x[2])))
    res_2[[l]][i,2] <- sum(apply(c_e, 1, function(x) c(x[1] <= y0 & y0 <= x[2])))
    len_2[[l]][i,1] <- sum(apply(c_m, 1, function(x) x[2]-x[1]))
    len_2[[l]][i,2] <- sum(apply(c_e, 1, function(x) x[2]-x[1]))
    nrw_2[[l]][i,1] <- nrow(c_m)
    nrw_2[[l]][i,2] <- nrow(c_e)
    
    conf.pred.ints <- lapply(data_list, function(z) conformal.pred.split(z$X, z$y, z$X0, alpha = alpha*(1-tau3),
                                                                         train.fun = funs$train, predict.fun = funs$predict))
    los <- lapply(conf.pred.ints, function(x) x$lo)
    los <- do.call(c, los)
    ups <- lapply(conf.pred.ints, function(x) x$up)
    ups <- do.call(c, ups)
    cis <- cbind(los, ups)
    c_m <- majority_vote(cis, rep(1/k, k), tau3)
    c_e <- exch_majority_vote(cis, tau3)
    
    res_3[[l]][i,1] <- sum(apply(c_m, 1, function(x) c(x[1] <= y0 & y0 <= x[2])))
    res_3[[l]][i,2] <- sum(apply(c_e, 1, function(x) c(x[1] <= y0 & y0 <= x[2])))
    len_3[[l]][i,1] <- sum(apply(c_m, 1, function(x) x[2]-x[1]))
    len_3[[l]][i,2] <- sum(apply(c_e, 1, function(x) x[2]-x[1]))
    nrw_3[[l]][i,1] <- nrow(c_m)
    nrw_3[[l]][i,2] <- nrow(c_e)
    
    if(i %% 50 == 0) cat("Iter: ", i, " (", k, ") ", "\n")
  }
}

save(res_1, res_2, res_3, len_1, len_2, len_3, nrw_1, nrw_2, nrw_3, file = "lasso_split.RData")
load("lasso_split.RData")

lapply(res_1, colMeans); lapply(res_2, colMeans); lapply(res_3, colMeans)
lapply(len_1, colMeans); lapply(len_2, colMeans); lapply(len_3, colMeans)
lapply(nrw_1, colMeans); lapply(nrw_2, colMeans); lapply(nrw_3, colMeans)

data_plot <- cbind(do.call(rbind, lapply(res_1, colMeans)), do.call(rbind, lapply(res_2, colMeans)), do.call(rbind, lapply(res_3, colMeans))) 
colnames(data_plot) <- c("cm1", "ce1", "cm2", "ce2", "cm3", "ce3")
data_plot <- as.data.frame(data_plot)
data_plot$k <- ks


# Plots -----
p1 <- ggplot(data_plot, aes(x = k)) +
  geom_line(aes(y = cm1, color = "tau=0.25", linetype = "cm")) +
  geom_line(aes(y = ce1, color = "tau=0.25", linetype = "ce")) +
  geom_line(aes(y = cm2, color = "tau=0.5", linetype = "cm")) +
  geom_line(aes(y = ce2, color = "tau=0.5", linetype = "ce")) +
  geom_line(aes(y = cm3, color = "tau=0.75", linetype = "cm")) +
  geom_line(aes(y = ce3, color = "tau=0.75", linetype = "ce")) +
  geom_point(aes(y = cm1, color = "tau=0.25")) +
  geom_point(aes(y = ce1, color = "tau=0.25")) +
  geom_point(aes(y = cm2, color = "tau=0.5"), shape = 18) +
  geom_point(aes(y = ce2, color = "tau=0.5"), shape = 18) +
  geom_point(aes(y = cm3, color = "tau=0.75"), shape = 15) +
  geom_point(aes(y = ce3, color = "tau=0.75"), shape = 15) +
  geom_hline(yintercept = 0.9, linetype = "dotted", color = "black") +
  scale_color_manual(
    values = c("tau=0.5" = "blue", "tau=0.25" = "red", "tau=0.75"="forestgreen"),
    labels = c(expression(tau==0.25), expression(tau==0.50), expression(tau==0.75)), name=expression(tau)) +
  scale_linetype_manual(
    values = c("cm" = "dashed", "ce" = "solid"), 
    labels = c(expression(C^E), expression(C^M)), name = "Meth.") +
  xlab("Number of splits") +
  ylab("Coverage") +
  theme_minimal() + ylim(0.9, 1) +
  theme(legend.position = "bottom", legend.text = element_text(size = 12))


data_plot <- cbind(do.call(rbind, lapply(len_1, colMeans)), do.call(rbind, lapply(len_2, colMeans)), do.call(rbind, lapply(len_3, colMeans))) 
colnames(data_plot) <- c("cm1", "ce1", "cm2", "ce2", "cm3", "ce3")
data_plot <- as.data.frame(data_plot)
data_plot$k <- ks
#data_plot <- data_plot[-1,]

p2 <- ggplot(data_plot, aes(x = k)) +
  geom_line(aes(y = cm1, color = "tau=0.25", linetype = "cm")) +
  geom_line(aes(y = ce1, color = "tau=0.25", linetype = "ce")) +
  geom_line(aes(y = cm2, color = "tau=0.5", linetype = "cm")) +
  geom_line(aes(y = ce2, color = "tau=0.5", linetype = "ce")) +
  geom_line(aes(y = cm3, color = "tau=0.75", linetype = "cm")) +
  geom_line(aes(y = ce3, color = "tau=0.75", linetype = "ce")) +
  geom_point(aes(y = cm1, color = "tau=0.25")) +
  geom_point(aes(y = ce1, color = "tau=0.25")) +
  geom_point(aes(y = cm2, color = "tau=0.5"), shape = 18) +
  geom_point(aes(y = ce2, color = "tau=0.5"), shape = 18) +
  geom_point(aes(y = cm3, color = "tau=0.75"), shape = 15) +
  geom_point(aes(y = ce3, color = "tau=0.75"), shape = 15) +
  scale_color_manual(
    values = c("tau=0.5" = "blue", "tau=0.25" = "red", "tau=0.75"="forestgreen"),
    labels = c(expression(tau==0.25), expression(tau==0.50), expression(tau==0.75)), name=expression(tau)) +
  scale_linetype_manual(
    values = c("cm" = "dashed", "ce" = "solid"), 
    labels = c(expression(C^E), expression(C^M)), name = "Meth.") +
  xlab("Number of splits") +
  ylab("Size") +
  theme_minimal()  +
  theme(legend.position = "bottom", legend.text = element_text(size = 12))

gridExtra::grid.arrange(p1, p2, ncol = 2)
