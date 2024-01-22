# WMA for high-dimensional regression
rm(list = ls())
library(conformalInference)
library(dplyr)
library(tidyverse)
library(glmnet)
library(ggplot2)
library(gridExtra)
library(limSolve)
source("utilsFuns.R")
source("/Users/gaspa/Desktop/Phd/Research/CMU/MergingSets/Code/utils.R")

# Functions -----
loss_fun <- function(c, sets, a, b){
  # c: target point
  # sets: list of set
  # a: parm
  size_int <- lapply(sets, function(x) x$up - x$lo) %>% unlist()
  cov_int  <- lapply(sets, function(x) c(x$lo <= c & c <= x$up)) %>% unlist()
  loss_vec <- a * size_int - b * cov_int
  return(loss_vec)
}

weighted_sum <- function(x, w){
  sum(x * w)
}

sizes <- function(sets){
  # sets: list of sets
  lapply(sets, function(x) x$up - x$lo) %>% unlist()
}


# Load data -----
dati <- read.table("affitti.txt", header = T)
dati <- dati %>% select(price, latitude, longitude, accommodates, bathrooms, bedrooms,
                        extra_people, minimum_nights, maximum_nights, number_of_reviews,
                        review_scores_rating, review_scores_accuracy, reviews_per_month,
                        aggressioni_1000, droga_1000, furti_1000)


# creation of the outcome and the X matrix
y <- log(dati$price)
X <- dati[,-1]
X <- scale(X)

set.seed(123)
train <- sample(1:NROW(y), 75000)
test  <- sample(setdiff(1:NROW(y), train), 75)

yt <- y[train]
Xt <- X[train,]

y0 <- y[test]
X0 <- X[test,]


# set the functions
nnet.funs <- list(
  train = function(x, y, ...) nnet.train(x, y, size = 8, decay = 1, linout = T, maxit = 2000),
  predict  = function(out, newx) nnet.preds(out, newx)
)

funs <- list()
funs[[1]] <- rf.funs(ntree = 250, varfrac = 4/16)
funs[[2]] <- lm.funs()
funs[[3]] <- lasso.funs(standardize = F, lambda = 0.1)
funs[[4]] <- ridge.funs(standardize = F, lambda = 0.1)
funs[[5]] <- nnet.funs

# Simulation ----
t     <- 75          # number of rounds
k     <- length(funs)# number of methods
w     <- rep(1, k)   # initial weights
a     <- 1           # lin. combination
b     <- 0           # lin. combination
alpha <- 0.1         # confidence level

#mat_resR <- mat_resM <- matrix(NA, ncol = 2, nrow = t)
mat_best <- rep(NA, t)
weights  <- losses <- matrix(NA, ncol = k, nrow = t)

set.seed(321)
for(i in 1:t){
  sel<- (1+1000*(i-1)):(1000*i) 
  Xa <- Xt[sel,]
  ya <- yt[sel]
  
  # prediction intervals
  conf.pred.ints <-lapply(funs, function(z) conformal.pred.split(Xa, ya, X0[i,], alpha = alpha/2,
                                                                 train.fun = z$train, predict.fun = z$predict))
  
  mat_best[i]  <- min(sizes(conf.pred.ints))
  # update weights
  loss        <- loss_fun(y0[i], conf.pred.ints, a, b)
  losses[i,]  <- loss
  
  if(i %% 10 == 0) cat(i, "\n")
} 


# Compute the algorithms -----
## Without NN ----
hed_alg1 <- hedge(losses[,1:4], 0.1)
hed_alg2 <- hedge(losses[,1:4], 1)
ada_alg  <- adahedge(losses[,1:4])

results1 <- data.frame(iter  = 1:length(losses[,1] %>% stats::filter(., rep(1/5,5))),
                      RF = losses[,1] %>% stats::filter(., rep(1/5,5)),
                      Hedge_0.1 = hed_alg1$h %>% stats::filter(., rep(1/5,5)),
                      Hedge_1   = hed_alg2$h %>% stats::filter(., rep(1/5,5)),
                      AdaHedge  = ada_alg$h%>% stats::filter(., rep(1/5,5)))


p1<-ggplot(results1, aes(x = iter)) +
  geom_line(aes(y = RF, color = "RF"), linetype = "solid", size = 1) +
  geom_line(aes(y = Hedge_0.1, color = "eta = 0.1"), linetype = "longdash", size = 1) +
  geom_line(aes(y = Hedge_1, color = "eta = 1"), linetype = "longdash", size = 1) +
  geom_line(aes(y = AdaHedge, color =  "Adaptive eta"), linetype = "dashed", size = 1) + 
  labs(title = "", x = "Iter", y = "Hedge Loss", color = "") +
  scale_color_manual(values = c("RF" = "blue", "eta = 0.1" = "red", "eta = 1" = "green", "Adaptive eta" = "orange")) +
  theme_minimal() + theme(legend.position = "bottom") +
  guides(linetype = guide_legend(keywidth = 3, keyheight = 1)) +
  ylim(1.55, 1.9)


## With NN -----
hed_alg1_nn <- hedge(losses, 0.1)
hed_alg2_nn <- hedge(losses, 1)
ada_alg_nn  <- adahedge(losses)

results2 <- data.frame(iter  = 1:length(losses[,1] %>% stats::filter(., rep(1/5,5))),
                      RF = losses[,1] %>% stats::filter(., rep(1/5,5)),
                      NN = losses[,5] %>% stats::filter(., rep(1/5,5)),
                      Hedge_0.1 = hed_alg1_nn$h %>% stats::filter(., rep(1/5,5)),
                      Hedge_1   = hed_alg2_nn$h %>% stats::filter(., rep(1/5,5)),
                      AdaHedge  = ada_alg_nn$h%>% stats::filter(., rep(1/5,5)))

p2<-ggplot(results2, aes(x = iter)) +
  geom_line(aes(y = RF, color = "RF"), linetype = "solid", size = 1) +
  geom_line(aes(y = NN, color = "NN"), linetype = "solid", size = 1) +
  geom_line(aes(y = Hedge_0.1, color = "eta = 0.1"), linetype = "longdash", size = 1) +
  geom_line(aes(y = Hedge_1, color = "eta = 1"), linetype = "longdash", size = 1) +
  geom_line(aes(y = AdaHedge, color = "Adaptive eta"), linetype = "dashed", size = 1) + 
  labs(title = "", x = "Iter", y = "Hedge Loss", color = "") +
  scale_color_manual(values = c("RF" = "blue", "eta = 0.1" = "red", "eta = 1" = "green", "Adaptive eta" = "orange", "NN" = "pink")) +
  theme_minimal() + theme(legend.position = "bottom") +
  ylim(1.55, 1.9)

grid.arrange(p1, p2, nrow = 1)

ada_alg$eta; ada_alg_nn$eta

data.eta <- data.frame("iter"= 2:t, "without.NN" = ada_alg$eta[2:t], "with.NN" = ada_alg_nn$eta[2:t])
p3<-ggplot(data.eta, aes(x = iter)) +
  geom_line(aes(y = without.NN, color = "K=4"), linetype = "solid", size = 1) +
  geom_line(aes(y = with.NN, color = "K=5"), linetype = "solid", size = 1) + 
  labs(title = "", x = "Iter", y = "eta", color = "") +
  geom_hline(yintercept = 0.1, color = "red", linetype = "dashed") + 
  geom_hline(yintercept = 1, color = "green", linetype = "dashed") +
  theme_minimal() + theme(legend.position = "bottom") 
p4<-ggplot(data.eta[2:t,], aes(x = iter)) +
  geom_line(aes(y = without.NN, color = "K=4"), linetype = "solid", size = 1) +
  geom_line(aes(y = with.NN, color = "K=5"), linetype = "solid", size = 1) +
  labs(title = "", x = "Iter", y = "eta", color = "") +
  geom_hline(yintercept = 0.1, color = "red", linetype = "dashed") + 
  geom_hline(yintercept = 1, color = "green", linetype = "dashed") +
  theme_minimal() + theme(legend.position = "bottom") 
grid.arrange(p3, p4, nrow = 1)

# Stacking (offline) -----
n_cross <- 10
mat_n   <- matrix(NA, nrow = NROW(Xt), ncol = k)
for(i in 1:n_cross){
  sel            <- (1+7500*(i-1)):(7500*i) 
  conf.pred.ints <- lapply(funs, function(z) conformal.pred.split(Xt[-sel,], yt[-sel], Xt[sel,], alpha = alpha/2, train.fun = z$train, predict.fun = z$predict))
  for(j in 1:k){
    mat_n[sel, j] <- conf.pred.ints[[j]]$pred
  }
  cat("Iter:", i, "\n")
}

E  <- rep(1, k)
Fv <- 1
G  <- diag(rep(1, k))
H  <- rep(0, k)

w_stack <- lsei(A = mat_n, B = yt, E = E, F = Fv, G = G, H = H)
w_stack 
h_stack <- data.frame(
  method = c("RF", "LM", "Lasso", "Ridge", "NN"),
  weight = w_stack$X
)

ggplot(h_stack, aes(x = method, y = weight)) +
  geom_bar(stat = "identity", position = "dodge") +
  labs(title = "Stacking", x = "", y = "Weight") +
  theme_minimal() + ylim(0, 1) + theme(legend.position = "bottom")

# weights -----
plots.w <- vector("list", 6)
num.t   <- c(1, 15, 30, 45, 60, 75)
for(i in num.t){
  data.p <- data.frame(
    algorithm = c(rep("eta = 0.1", 4), rep("eta = 1", 4), rep("Adaptive eta", 4)),
    method = rep(c("RF", "LM", "Lasso", "Ridge"), 3),
    values = c(hed_alg1$weights[i,], hed_alg2$weights[i,], ada_alg$weights[i,])
  )
  j <- ifelse(i==1, 1, i/15 + 1)
  plots.w[[j]] <- ggplot(data.p, aes(x = method, y = values, fill = algorithm)) +
    geom_bar(stat = "identity", position = "dodge") +
    facet_grid(. ~ algorithm, scales = "free", space = "free") +
    labs(title = paste("t =",i), x = "", y = "Weight") +
    theme_minimal() + ylim(0, 1) + theme(legend.position = "none") +
    theme(axis.text.x = element_text(angle = 90, hjust = 0.5, vjust = 0.5))
}

grid.arrange(plots.w[[1]], plots.w[[2]], plots.w[[3]], plots.w[[4]], plots.w[[5]], plots.w[[6]], nrow = 2)


plots.w <- vector("list", 6)
num.t   <- c(1, 15, 30, 45, 60, 75)
for(i in num.t){
   data.p <- data.frame(
     algorithm = c(rep("eta = 0.1", 5), rep("eta = 1", 5), rep("Adaptive eta", 5)),
     method = rep(c("RF", "LM", "Lasso", "Ridge", "NN"), 3),
     values = c(hed_alg1_nn$weights[i,], hed_alg2_nn$weights[i,], ada_alg_nn$weights[i,])
   )
   j <- ifelse(i==1, 1, i/15 + 1)
   plots.w[[j]] <- ggplot(data.p, aes(x = method, y = values, fill = algorithm)) +
     geom_bar(stat = "identity", position = "dodge") +
     facet_grid(. ~ algorithm, scales = "free", space = "free") +
     labs(title = paste("t =",i), x = "", y = "Weight") +
     theme_minimal() + ylim(0, 1) + theme(legend.position = "none") +
     theme(axis.text.x = element_text(angle = 90, hjust = 0.5, vjust = 0.5))
}

grid.arrange(plots.w[[1]], plots.w[[2]], plots.w[[3]], plots.w[[4]], plots.w[[5]], plots.w[[6]], nrow = 2)
