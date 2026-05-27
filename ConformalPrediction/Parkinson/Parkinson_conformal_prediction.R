rm(list = ls())
library(dplyr)
library(ggplot2)
library(glmnet)
library(plyr)
library(lars)
library(randomForest)
library(conformalInference)
library(corrplot)
source("utils.R")

# Load data -----
dati <- read.table("merged_dataset.csv", header = T, sep = ",")
head(dati)

# creation of the outcome and the X matrix
y <- dati$total_UPDRS
X <- dati[,-c(1,2)]
X[,1:18] <- scale(X[,1:18])

set.seed(1234)
train <- sample(1:NROW(y), 5000)
test  <- setdiff(1:NROW(y), train)

ytrain <- y[train]
Xtrain <- X[train,]

y0 <- y[test]
X0 <- X[test,]

# Analysis ----
ggplot(as.data.frame(ytrain), aes(x = ytrain)) +
  geom_histogram(fill = "lightblue", color = "black", bins = 10) +
  labs(title = "Histogram of total UPDRS", x = "total UPDRS", y = "Freq")

corrplot(cor(Xtrain))

matr_corr <- cor(Xtrain)
matr_corr[lower.tri(matr_corr, diag=T)] <- NA
correlations <- data.frame(var1 = rep(colnames(matr_corr), ncol(matr_corr)),
                           var2 = rep(rownames(matr_corr), each = nrow(matr_corr)),
                           cor = as.vector(matr_corr))
correlations <- na.omit(correlations)
correlations %>% filter(cor < -0.975 | cor > 0.975)

vars_to_rem <- which(colnames(Xtrain) %in% c("Jitter.RAP", "Jitter.DDP", "Shimmer", "Shimmer.dB.", "Shimmer.APQ3"))
Xtrain <- Xtrain[, -vars_to_rem]
X0 <- X0[, -vars_to_rem]
corrplot(cor(Xtrain))


nnet.funs <- list(
  train = function(x, y, ...) nnet.train(x, y, size = 20, decay = 1, linout = T, maxit = 2000),
  predict  = function(out, newx) nnet.preds(out, newx)
)


# Conformal Predictions -----
funs <- list()
funs[[1]] <- lm.funs()
funs[[2]] <- lasso.funs(standardize = F, cv = T, cv.rule = "min")
funs[[3]] <- rf.funs(ntree = 400)
funs[[4]] <- nnet.funs

conf.ints1 <- lapply(funs, function(z) conformal.pred.split(Xtrain, ytrain, as.matrix(X0), alpha = 0.1,
                                                            train.fun = z$train, predict.fun = z$predict))


# Coverage of Majority Vote and Randomized Majority Vote
out        <- lapply(conf.ints1, function(x) c(x$lo <= y0 & y0 <= x$up))
res_out   <- NULL
for(i in 1:length(out)){
  res_out <- cbind(res_out, out[[i]])
}
colMeans(res_out)

set.seed(1234)
our_out1 <- rowMeans(res_out) > (1/2)
u1 <- runif(NROW(res_out), 0, 1/2)
our_out2 <- rowMeans(res_out) > (1/2 + u1)
u2 <- runif(NROW(res_out))
our_out3 <- rowMeans(res_out) > u2
mean(our_out1); mean(our_out2);  mean(our_out3)
coverages <- c(colMeans(res_out), mean(our_out1), mean(our_out2), mean(our_out3))

# Lenght of the methods
avg_length <- lapply(conf.ints1, function(x) (x$up - x$lo) %>% mean)

res_len <- res_dou <- matrix(NA, nrow = length(y0), ncol = 4)
cov_e <- rep(0, length(y0))
for(i in 1:length(y0)){
  M <- matrix(NA, nrow = 4, ncol = 2)
  for(j in 1:4){
    M[j,] <- c(conf.ints1[[j]]$lo[i], conf.ints1[[j]]$up[i])
  }
  ci1 <- as.matrix(majority_vote(M, rep(1/4, 4), 0.5))
  res_dou[i,1] <- nrow(ci1)
  for(l in 1:nrow(ci1)){
    res_len[i,1] <- ifelse(is.na(ci1[1,1]), 0, ci1[l,2]-ci1[l,1])
  }
  
  ci2 <- as.matrix(majority_vote(M, rep(1/4, 4), 0.5 + u1[i]))
  res_dou[i,2] <- nrow(ci2)
  for(l in 1:nrow(ci2)){
    res_len[i,2] <- ifelse(is.na(ci2[1,1]), 0, ci2[l,2]-ci2[l,1])
  }
  
  ci3 <- as.matrix(majority_vote(M, rep(1/4, 4), u2[i]))
  res_dou[i,3] <- nrow(ci3)
  for(l in 1:nrow(ci3)){
    res_len[i,3] <- ifelse(is.na(ci3[1,1]), 0, ci3[l,2]-ci3[l,1])
  }
  
  ci4 <- as.matrix(exch_majority_vote(M))
  res_dou[i,4] <- nrow(ci4)
  for(l in 1:nrow(ci4)){
    res_len[i,4] <- ifelse(is.na(ci4[1,1]), 0, ci4[l,2]-ci4[l,1])
    cov_e[i] <- ifelse(is.na(ci4[1,1]), 0, cov_e[i] + I(ci4[l,1]<= y0[i] & y0[i] <= ci4[l,2]))
  }
}

coverages   <- c(coverages, mean(cov_e))
lengths     <- c(avg_length%>%unlist, colMeans(res_len))
colMeans(res_dou>1)*100

# Table -----
methods_name <- c("Linear Model", "Lasso", "Random Forest", "Neural Net", "Majority Vote", "Randomized Majority Vote", "Randomized Vote", "Exch. Vote")
tab_p <- data.frame("Methods" = methods_name, 
                    "Coverage" = coverages,
                    "Lengths" = lengths)
xtable::xtable(t(tab_p), digits = 3)

