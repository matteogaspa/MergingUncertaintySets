rm(list = ls())
library(glmnet)
library(ggplot2)
library(dplyr)
library(gridExtra)
source("utils_elec2.R")

# Load data -----
elec <- read.csv("electricity-normalized.csv")
head(elec)
str(elec)
# delete constant transfer
const.ind <- min(which(elec$transfer != elec$transfer[1]))
elec <- elec[const.ind:NROW(elec),]
const.ind <- min(which(elec$transfer != elec$transfer[1]))
elec <- elec[const.ind:NROW(elec),]
# from 9 to 12 am
time.ran <- c(9*2/48, 0.5)
elec <- elec[(elec$period >= time.ran[1] & elec$period <= time.ran[2]),]
elec <- elec[,4:8]

# SECTION 1 -----
## lm -----
runlmElecPred <- function(Y,X,alpha,gamma,tinit = 445,splitSize = 0.75,updateMethod = "Simple",momentumBW=0.95){
  T <- length(Y)
  ## Initialize data storage variables
  alphaTrajectory <- rep(alpha,T-tinit+1)
  adaptErrSeq <-  rep(0,T-tinit+1)
  noAdaptErrorSeq <-  rep(0,T-tinit+1)
  alphat <- alpha
  piAdapt <- matrix(NA, nrow=T-tinit+1, ncol=2)
  piNoAdapt <- matrix(NA, nrow=T-tinit+1, ncol=2)
  for(t in tinit:T){
    ### Split data into training and calibration set
    trainPoints <- sample(1:(tinit-1),round(splitSize*tinit))
    calpoints <- (1:(tinit-1))[-trainPoints]
    newX <- X[(t-tinit+1):(t-1),]
    newY <- Y[(t-tinit+1):(t-1)]
    Xtrain <- newX[trainPoints,]
    Ytrain <- newY[trainPoints]
    XCal <- newX[calpoints,]
    YCal <- newY[calpoints]
    
    ### Fit regression on training setting
    lmfit <- lm(Ytrain ~ Xtrain)
    
    ### Compute conformity score on calibration set and on new data example
    predForCal <- cbind(rep(1,nrow(XCal)),XCal)%*%lmfit$coef
    scores <- abs(predForCal - YCal)
    predt <- as.numeric(c(1, X[t,])%*%lmfit$coef)
    newScore <- abs(predt - Y[t])
    
    ### Compute errt for both methods
    confQuantNaive <- quantile(scores,1-alpha)
    piNoAdapt[t-tinit+1,] <- c(predt - confQuantNaive, predt + confQuantNaive)
    noAdaptErrorSeq[t-tinit+1] <- as.numeric(confQuantNaive < newScore)
    if(alphat >=1){
      adaptErrSeq[t-tinit+1] <- 1
      confQuantAdapt <- 0
      piAdapt[t-tinit+1,] <- c(predt - confQuantAdapt, predt + confQuantAdapt)
    }else if (alphat <=0){
      adaptErrSeq[t-tinit+1] <- 0
      confQuantAdapt <- Inf
      piAdapt[t-tinit+1,] <- c(predt - confQuantAdapt, predt + confQuantAdapt)
    }else{
      confQuantAdapt <- quantile(scores,probs=1-alphat)
      piAdapt[t-tinit+1,] <- c(predt - confQuantAdapt, predt + confQuantAdapt)
      adaptErrSeq[t-tinit+1] <- as.numeric(confQuantAdapt < newScore)
    }
    
    ## update alphat
    alphaTrajectory[t-tinit+1] <- alphat
    if(updateMethod=="Simple"){
      alphat <- alphat + gamma*(alpha-adaptErrSeq[t-tinit+1])
    }else if(updateMethod=="Momentum"){
      w <- rev(momentumBW^(1:(t-tinit+1)))
      w <- w/sum(w)
      alphat <- alphat + gamma*(alpha - sum(adaptErrSeq[1:(t-tinit+1)]*w))
    }
    
    if(t %% 100 == 0){
      print(sprintf("Done %i time steps",t))
    }
  }
  return(list(alpha_t=alphaTrajectory,
              AdaptErr=adaptErrSeq,
              noAdaptErr=noAdaptErrorSeq,
              piAdapt=piAdapt,
              piNoAdapt=piNoAdapt))
}

lmElecPred <- runlmElecPred(elec$transfer, as.matrix(elec[,1:4]), alpha = 0.05, gamma = 0.005)

set.seed(12345)
data.plot <- data.frame(
  iter = 1:3000,
  Adapt = 1 - lmElecPred$AdaptErr %>% stats::filter(., rep(1/400, 400)),
  noAdapt = 1 - lmElecPred$noAdaptErr %>% stats::filter(., rep(1/400, 400)),
  Bernoulli = rbinom(3000, 1, 0.95) %>% stats::filter(., rep(1/400, 400))
)


plm<-ggplot(data.plot, aes(x = iter)) +
  geom_line(aes(y = Adapt, color = "Adapt"), linetype = "solid") +
  geom_line(aes(y = noAdapt, color = "No Adapt"), linetype = "solid") +
  geom_line(aes(y = Bernoulli, color = "Bern 0.95"), linetype = "solid") +
  geom_hline(yintercept = 0.95, color = "black", linetype = "dashed") +
  labs(title = "LM", x = "Iter", y = "Local Level Coverage", color = "") +
  scale_color_manual(values = c("Adapt" = "blue", "No Adapt" = "red", "Bern 0.95" = "gray")) +
  theme_minimal() + theme(legend.position = "bottom") +
  ylim(0.7, 1)
plm

## ridge ----
runRidgeElecPred <- function(Y,X,alpha,gamma,tinit = 445,splitSize = 0.75,updateMethod = "Simple",momentumBW=0.95,lambda=0.01){
  T <- length(Y)
  ## Initialize data storage variables
  alphaTrajectory <- rep(alpha,T-tinit+1)
  adaptErrSeq <-  rep(0,T-tinit+1)
  noAdaptErrorSeq <-  rep(0,T-tinit+1)
  alphat <- alpha
  piAdapt <- matrix(NA, nrow=T-tinit+1, ncol=2)
  piNoAdapt <- matrix(NA, nrow=T-tinit+1, ncol=2)
  for(t in tinit:T){
    ### Split data into training and calibration set
    trainPoints <- sample(1:(tinit-1),round(splitSize*tinit))
    calpoints <- (1:(tinit-1))[-trainPoints]
    newX <- X[(t-tinit+1):(t-1),]
    newY <- Y[(t-tinit+1):(t-1)]
    Xtrain <- newX[trainPoints,]
    Ytrain <- newY[trainPoints]
    XCal <- newX[calpoints,]
    YCal <- newY[calpoints]
    
    ### Fit ridge regression on training setting
    mfit <- glmnet(Xtrain, Ytrain, lambda = lambda, standardize = FALSE, intercept = TRUE, alpha = 0)
    beta <- as.vector(coef(mfit))
    
    ### Compute conformity score on calibration set and on new data example
    predForCal <- cbind(rep(1,nrow(XCal)),XCal)%*%beta
    scores <- abs(predForCal - YCal)
    predt <- as.numeric((c(1, X[t,]))%*%beta)
    newScore <- abs(predt - Y[t])
    
    ## Compute errt for both methods
    confQuantNaive <- quantile(scores,1-alpha)
    piNoAdapt[t-tinit+1,] <- c(predt - confQuantNaive, predt + confQuantNaive)
    noAdaptErrorSeq[t-tinit+1] <- as.numeric(confQuantNaive < newScore)
    if(alphat >=1){
      adaptErrSeq[t-tinit+1] <- 1
      confQuantAdapt <- 0
      piAdapt[t-tinit+1,] <- c(predt - confQuantAdapt, predt + confQuantAdapt)
    }else if (alphat <=0){
      adaptErrSeq[t-tinit+1] <- 0
      confQuantAdapt <- Inf
      piAdapt[t-tinit+1,] <- c(predt - confQuantAdapt, predt + confQuantAdapt)
    }else{
      confQuantAdapt <- quantile(scores,probs=1-alphat)
      piAdapt[t-tinit+1,] <- c(predt - confQuantAdapt, predt + confQuantAdapt)
      adaptErrSeq[t-tinit+1] <- as.numeric(confQuantAdapt < newScore)
    }
    
    ## update alphat
    alphaTrajectory[t-tinit+1] <- alphat
    if(updateMethod=="Simple"){
      alphat <- alphat + gamma*(alpha-adaptErrSeq[t-tinit+1])
    }else if(updateMethod=="Momentum"){
      w <- rev(momentumBW^(1:(t-tinit+1)))
      w <- w/sum(w)
      alphat <- alphat + gamma*(alpha - sum(adaptErrSeq[1:(t-tinit+1)]*w))
    }
    
    if(t %% 100 == 0){
      print(sprintf("Done %i time steps",t))
    }
  }
  return(list(alpha_t=alphaTrajectory,
              AdaptErr=adaptErrSeq,
              noAdaptErr=noAdaptErrorSeq,
              piAdapt=piAdapt,
              piNoAdapt=piNoAdapt))
}

RidgeElecPred <- runRidgeElecPred(elec$transfer, as.matrix(elec[,1:4]), alpha = 0.05, gamma = 0.005)

set.seed(12345)
data.plot <- data.frame(
  iter = 1:3000,
  Adapt = 1 - RidgeElecPred$AdaptErr %>% stats::filter(., rep(1/400, 400)),
  noAdapt = 1 - RidgeElecPred$noAdaptErr %>% stats::filter(., rep(1/400, 400)),
  Bernoulli = rbinom(3000, 1, 0.95) %>% stats::filter(., rep(1/400, 400))
)


pRidge<-ggplot(data.plot, aes(x = iter)) +
  geom_line(aes(y = Adapt, color = "Adapt"), linetype = "solid") +
  geom_line(aes(y = noAdapt, color = "No Adapt"), linetype = "solid") +
  geom_line(aes(y = Bernoulli, color = "Bern 0.95"), linetype = "solid") +
  geom_hline(yintercept = 0.95, color = "black", linetype = "dashed") +
  labs(title = "Ridge", x = "Iter", y = "Local Level Coverage", color = "") +
  scale_color_manual(values = c("Adapt" = "blue", "No Adapt" = "red", "Bern 0.95" = "gray")) +
  theme_minimal() + theme(legend.position = "bottom") +
  ylim(0.7, 1)
pRidge


## lasso -----
runLassoElecPred <- function(Y,X,alpha,gamma,tinit = 445,splitSize = 0.75,updateMethod = "Simple",momentumBW=0.95, lambda=0.01){
  T <- length(Y)
  ## Initialize data storage variables
  alphaTrajectory <- rep(alpha,T-tinit+1)
  adaptErrSeq <-  rep(0,T-tinit+1)
  noAdaptErrorSeq <-  rep(0,T-tinit+1)
  alphat <- alpha
  piAdapt <- matrix(NA, nrow=T-tinit+1, ncol=2)
  piNoAdapt <- matrix(NA, nrow=T-tinit+1, ncol=2)
  for(t in tinit:T){
    ### Split data into training and calibration set
    trainPoints <- sample(1:(tinit-1),round(splitSize*tinit))
    calpoints <- (1:(tinit-1))[-trainPoints]
    newX <- X[(t-tinit+1):(t-1),]
    newY <- Y[(t-tinit+1):(t-1)]
    Xtrain <- newX[trainPoints,]
    Ytrain <- newY[trainPoints]
    XCal <- newX[calpoints,]
    YCal <- newY[calpoints]
    
    ### Fit lasso regression on training setting
    mfit <- glmnet(Xtrain, Ytrain, lambda = lambda, standardize = FALSE, intercept = TRUE, alpha = 1)
    beta <- as.vector(coef(mfit))
    ### Compute conformity score on calibration set and on new data example
    predForCal <- cbind(rep(1,nrow(XCal)),XCal)%*%beta
    scores <- abs(predForCal - YCal)
    predt <- as.numeric((c(1, X[t,]))%*%beta)
    newScore <- abs(predt - Y[t])
    
    ## Compute errt for both methods
    confQuantNaive <- quantile(scores,1-alpha)
    piNoAdapt[t-tinit+1,] <- c(predt - confQuantNaive, predt + confQuantNaive)
    noAdaptErrorSeq[t-tinit+1] <- as.numeric(confQuantNaive < newScore)
    if(alphat >=1){
      adaptErrSeq[t-tinit+1] <- 1
      confQuantAdapt <- 0
      piAdapt[t-tinit+1,] <- c(predt - confQuantAdapt, predt + confQuantAdapt)
    }else if (alphat <=0){
      adaptErrSeq[t-tinit+1] <- 0
      confQuantAdapt <- Inf
      piAdapt[t-tinit+1,] <- c(predt - confQuantAdapt, predt + confQuantAdapt)
    }else{
      confQuantAdapt <- quantile(scores,probs=ifelse(alphat>0, 1-alphat, 1))
      piAdapt[t-tinit+1,] <- c(predt - confQuantAdapt, predt + confQuantAdapt)
      adaptErrSeq[t-tinit+1] <- as.numeric(confQuantAdapt < newScore)
    }
    
    ## update alphat
    alphaTrajectory[t-tinit+1] <- alphat
    if(updateMethod=="Simple"){
      alphat <- alphat + gamma*(alpha-adaptErrSeq[t-tinit+1])
    }else if(updateMethod=="Momentum"){
      w <- rev(momentumBW^(1:(t-tinit+1)))
      w <- w/sum(w)
      alphat <- alphat + gamma*(alpha - sum(adaptErrSeq[1:(t-tinit+1)]*w))
    }
    
    if(t %% 100 == 0){
      print(sprintf("Done %i time steps",t))
    }
  }
  return(list(alpha_t=alphaTrajectory,
              AdaptErr=adaptErrSeq,
              noAdaptErr=noAdaptErrorSeq,
              piAdapt=piAdapt,
              piNoAdapt=piNoAdapt))
}

LassoElecPred <- runLassoElecPred(elec$transfer, as.matrix(elec[,1:4]), alpha = 0.05, gamma = 0.005)

set.seed(12345)
data.plot <- data.frame(
  iter = 1:3000,
  Adapt = 1 - LassoElecPred$AdaptErr %>% stats::filter(., rep(1/400, 400)),
  noAdapt = 1 - LassoElecPred$noAdaptErr %>% stats::filter(., rep(1/400, 400)),
  Bernoulli = rbinom(3000, 1, 0.95) %>% stats::filter(., rep(1/400, 400))
)



pLasso<-ggplot(data.plot, aes(x = iter)) +
  geom_line(aes(y = Adapt, color = "Adapt"), linetype = "solid") +
  geom_line(aes(y = noAdapt, color = "No Adapt"), linetype = "solid") +
  geom_line(aes(y = Bernoulli, color = "Bern 0.95"), linetype = "solid") +
  geom_hline(yintercept = 0.95, color = "black", linetype = "dashed") +
  labs(title = "Lasso", x = "Iter", y = "Local Level Coverage", color = "") +
  scale_color_manual(values = c("Adapt" = "blue", "No Adapt" = "red", "Bern 0.95" = "gray")) +
  theme_minimal() + theme(legend.position = "bottom") +
  ylim(0.7, 1)

my.plot1 <- grid.arrange(plm, pRidge, pLasso, nrow = 1)
my.plot1
LassoElecPred$alpha_t
LassoElecPred$piAdapt

## COMA -----
### Adaptive eta
loss.matrix <- cbind(lmElecPred$piAdapt[,2] - lmElecPred$piAdapt[,1], 
                     RidgeElecPred$piAdapt[,2] - lmElecPred$piAdapt[,1],
                     LassoElecPred$piAdapt[,2] - LassoElecPred$piAdapt[,1])
loss.matrix2 <- pgamma(loss.matrix, shape = 0.1, rate = 0.1)
adahedAlg <- adahedge(loss.matrix2)


N <- NROW(adahedAlg$weights)
K <- NCOL(adahedAlg$weights)
t <- NROW(elec)-N
pi.m <- pi.wm <- matrix(NA, N, 2)
m.cov <- rep(NA, N)
wm.cov <- rep(NA, N)

for(i in 1:N){
  conf.t <- rbind(lmElecPred$piAdapt[i,],
                  RidgeElecPred$piAdapt[i,],
                  LassoElecPred$piAdapt[i,])
  maj.int <- majority_vote(conf.t, w=adahedAlg$weights[i,], 0.5)
  wmaj.int <- majority_vote(conf.t, w=adahedAlg$weights[i,], runif(1, 0.5, 1))
  pi.m[i,] <- maj.int
  pi.wm[i,] <- wmaj.int
  m.cov[i] <- as.numeric((maj.int[1] <= elec$transfer[t+i]) & (elec$transfer[t+i] <= maj.int[2]))
  wm.cov[i] <- as.numeric((wmaj.int[1] <= elec$transfer[t+i]) &(elec$transfer[t+i] <= wmaj.int[2]))
  if(i %% 100 == 0){
    print(sprintf("Done %i time steps",i))
  }
}

data.plot <- data.frame(
  iter = 1:N,
  local.m = stats::filter(m.cov, rep(1/400, 400)),
  local.wm = stats::filter(wm.cov, rep(1/400, 400)),
  bern = stats::filter(rbinom(N, 1, 0.9), rep(1/400, 400))
)


pM<-ggplot(data.plot, aes(x = iter)) +
  geom_line(aes(y = local.m, color = "Weighted Majority"), linetype = "solid") +
  geom_line(aes(y = local.wm, color = "Rand. Weighted Majority"), linetype = "solid") +
  geom_line(aes(y = bern, color = "Bern 0.9"), linetype = "solid") +
  geom_hline(yintercept = 0.90, color = "black", linetype = "dashed") +
  labs(title = "AdaHedge", x = "Iter", y = "Local Level Coverage", color = "") +
  scale_color_manual(values = c("Weighted Majority" = "forestgreen", "Rand. Weighted Majority" = "cyan3", "Bern 0.9" = "orange")) +
  theme_minimal() + theme(legend.position = "bottom")+
  ylim(0.7,1)
pM

grid.arrange(plm, pRidge, pLasso, pM, nrow = 2, ncol = 2)


### Fixed eta
hedAlg <- hedge(loss.matrix2, eta = 0.1)
N <- NROW(adahedAlg$weights)
K <- NCOL(adahedAlg$weights)
t <- NROW(elec)-N
pi.m <- pi.wm <- matrix(NA, N, 2)
m.cov <- rep(NA, N)
wm.cov <- rep(NA, N)

for(i in 1:N){
  conf.t <- rbind(lmElecPred$piAdapt[i,],
                  RidgeElecPred$piAdapt[i,],
                  LassoElecPred$piAdapt[i,])
  maj.int <- majority_vote(conf.t, w=hedAlg$weights[i,], 0.5)
  wmaj.int <- majority_vote(conf.t, w=hedAlg$weights[i,], runif(1,0.5,1))
  pi.m[i,] <- maj.int
  pi.wm[i,] <- wmaj.int
  m.cov[i] <- as.numeric((maj.int[1] <= elec$transfer[t+i]) & (elec$transfer[t+i] <= maj.int[2]))
  wm.cov[i] <- as.numeric((wmaj.int[1] <= elec$transfer[t+i]) &(elec$transfer[t+i] <= wmaj.int[2]))
  if(i %% 100 == 0){
    print(sprintf("Done %i time steps",i))
  }
}

wm.cov[is.na(wm.cov)] <- 0

data.plot <- data.frame(
  iter = 1:N,
  local.m = stats::filter(m.cov, rep(1/400, 400)),
  local.wm = stats::filter(wm.cov, rep(1/400, 400)),
  bern = stats::filter(rbinom(N, 1, 0.9), rep(1/400, 400))
)


pM2<-ggplot(data.plot, aes(x = iter)) +
  geom_line(aes(y = local.m, color = "Weighted Majority"), linetype = "solid") +
  geom_line(aes(y = local.wm, color = "Rand. Weighted Majority"), linetype = "solid") +
  geom_line(aes(y = bern, color = "Bern 0.9"), linetype = "solid") +
  geom_hline(yintercept = 0.90, color = "black", linetype = "dashed") +
  labs(title = "Hedge (eta=0.1)", x = "Iter", y = "Local Level Coverage", color = "") +
  scale_color_manual(values = c("Weighted Majority" = "forestgreen", "Rand. Weighted Majority" = "cyan3", "Bern 0.9" = "orange")) +
  theme_minimal() + theme(legend.position = "bottom")+
  ylim(0.7,1)
pM2

my.plot2 <- grid.arrange(pM, pM2, ncol = 2)

grid.arrange(my.plot1, my.plot2, nrow = 2)
adahedAlg$weights

ts.plot(hedAlg$weights, col=1:3)


# SECTION 2 ------
set.seed(123)
runMix <- function(Y, X, alpha, gamma, tinit = 445, splitSize = 0.75, lambda = 0.01, nu = 0.1){
  T <- length(Y)
  ## Initialize data storage variables
  alphaTrajectory <- rep(alpha,T-tinit+1)
  adaptErrSeq <-  rep(1,T-tinit+1)
  noAdaptErrorSeq <-  rep(1,T-tinit+1)
  alphat <- alpha/2
  #piAdapt <- matrix(NA, nrow=T-tinit+1, ncol=2)
  #piNoAdapt <- matrix(NA, nrow=T-tinit+1, ncol=2)
  wTrajectory <- matrix(NA, nrow=T-tinit+1, ncol=3)
  wt <- rep(1/3, 3)
  for(t in tinit:T){
    ### Split data into training and calibration set
    trainPoints <- sample(1:(tinit-1),round(splitSize*tinit))
    calpoints <- (1:(tinit-1))[-trainPoints]
    newX <- X[(t-tinit+1):(t-1),]
    newY <- Y[(t-tinit+1):(t-1)]
    Xtrain <- newX[trainPoints,]
    Ytrain <- newY[trainPoints]
    XCal <- newX[calpoints,]
    YCal <- newY[calpoints]
    
    ### Fit regression algorithms on training setting
    mfit1 <- lm(Ytrain ~ Xtrain)
    mfit2 <- glmnet(Xtrain, Ytrain, lambda = lambda, standardize = FALSE, intercept = TRUE, alpha = 0)
    mfit3 <- glmnet(Xtrain, Ytrain, lambda = lambda, standardize = FALSE, intercept = TRUE, alpha = 1)
    
    ### Compute conformity score on calibration set and on new data example for model 1
    predForCal1 <- cbind(rep(1,nrow(XCal)),XCal)%*%mfit1$coef
    scores1 <- abs(predForCal1 - YCal)
    predt1 <- as.numeric(c(1, X[t,])%*%mfit1$coef)

    ### Compute conformity score on calibration set and on new data example for model 2
    beta2 <- as.vector(coef(mfit2))
    predForCal2 <- cbind(rep(1,nrow(XCal)),XCal)%*%beta2
    scores2 <- abs(predForCal2 - YCal)
    predt2 <- as.numeric((c(1, X[t,]))%*%beta2)

    ### Compute conformity score on calibration set and on new data example for model 3
    beta3 <- as.vector(coef(mfit3))
    predForCal3 <- cbind(rep(1,nrow(XCal)),XCal)%*%beta3
    scores3 <- abs(predForCal3 - YCal)
    predt3 <- as.numeric((c(1, X[t,]))%*%beta3)

    ### Compute errt for both methods
    confQuantnoAdapt1 <- quantile(scores1,probs=1-alpha)
    confQuantnoAdapt2 <- quantile(scores2,probs=1-alpha)
    confQuantnoAdapt3 <- quantile(scores3,probs=1-alpha)
    confQuantsnoAdapt  <- matrix(NA, nrow = 3, ncol = 2)
    confQuantsnoAdapt[1,] <- c(predt1 - confQuantnoAdapt1, predt1 + confQuantnoAdapt1)
    confQuantsnoAdapt[2,] <- c(predt2 - confQuantnoAdapt2, predt2 + confQuantnoAdapt2)
    confQuantsnoAdapt[3,] <- c(predt3 - confQuantnoAdapt3, predt3 + confQuantnoAdapt3)
    
    piNoAdapt <- majority_vote(confQuantsnoAdapt, rep(1,3), 0.5)
    for(j in nrow(piNoAdapt)){
      noAdaptErrorSeq[t-tinit+1] <- 1 -noAdaptErrorSeq[t-tinit+1]*(as.numeric(piNoAdapt[j,1] < Y[t] && Y[t] < piNoAdapt[j,2]))
    }
    
    if(alphat >=1){
      adaptErrSeq[t-tinit+1] <- 1
      confQuantAdapt <- 0
      #piAdapt[t-tinit+1,] <- c(predt1 - confQuantnoAdapt1, predt1 + confQuantnoAdapt1)
      wTrajectory[t-tinit+1,] <- wt/sum(wt)
    }else if (alphat <=0){
      adaptErrSeq[t-tinit+1] <- 0
      confQuantAdapt <- Inf
      #piAdapt[t-tinit+1,] <- c(predt1 - confQuantnoAdapt1, predt1 + confQuantnoAdapt1)
      wTrajectory[t-tinit+1,] <- wt/sum(wt)
    }else{
      confQuantAdapt1 <- quantile(scores1,probs=1-alphat)
      confQuantAdapt2 <- quantile(scores2,probs=1-alphat)
      confQuantAdapt3 <- quantile(scores3,probs=1-alphat)
      confQuants  <- matrix(NA, nrow = 3, ncol = 2)
      confQuants[1,] <- c(predt1 - confQuantAdapt1, predt1 + confQuantAdapt1)
      confQuants[2,] <- c(predt2 - confQuantAdapt2, predt2 + confQuantAdapt2)
      confQuants[3,] <- c(predt3 - confQuantAdapt3, predt3 + confQuantAdapt3)
      piAdapt <- majority_vote(confQuants, rep(1,3), 0.5)
      for(j in nrow(piNoAdapt)){
        adaptErrSeq[t-tinit+1] <- 1 - adaptErrSeq[t-tinit+1]*(as.numeric(piAdapt[j,1] < Y[t] && Y[t] < piAdapt[j,2]))
      }
      ## update weights
      losst <- pgamma(confQuants[,2]-confQuants[,1], 0.1, 0.1)
      wTrajectory[t-tinit+1,] <- wt/sum(wt)
      wt <- wt * exp(-nu*losst)
    }
    
    ## update alphat
    alphaTrajectory[t-tinit+1] <- alphat
    alphat <- alphat + gamma*(alpha-adaptErrSeq[t-tinit+1])
    
    if(t %% 100 == 0){
      print(sprintf("Done %i time steps",t))
    }
  }
  return(list(alpha_t=alphaTrajectory,
              AdaptErr=adaptErrSeq,
              noAdaptErr=noAdaptErrorSeq,
              piAdapt=piAdapt,
              piNoAdapt=piNoAdapt,
              weights=wTrajectory))
}

rMix <- runMix(elec$transfer, as.matrix(elec[,1:4]), alpha = 0.05, gamma = 0.005)

runMixAdapt <- function(Y, X, alpha, gamma, tinit = 445, splitSize = 0.75, lambda = 0.01){
  T <- length(Y)
  ## Initialize data storage variables
  alphaTrajectory <- rep(alpha,T-tinit+1)
  adaptErrSeq <-  rep(1,T-tinit+1)
  noAdaptErrorSeq <-  rep(1,T-tinit+1)
  alphat <- alpha
  piAdapt <- matrix(NA, nrow=T-tinit+1, ncol=2)
  piNoAdapt <- matrix(NA, nrow=T-tinit+1, ncol=2)
  wTrajectory <- matrix(NA, nrow=T-tinit+1, ncol=3)
  lossIter <- matrix(NA, nrow=T-tinit+1, ncol=3)
  wt <- rep(1/3, 3)
  for(t in tinit:T){
    ### Split data into training and calibration set
    trainPoints <- sample(1:(tinit-1),round(splitSize*tinit))
    calpoints <- (1:(tinit-1))[-trainPoints]
    newX <- X[(t-tinit+1):(t-1),]
    newY <- Y[(t-tinit+1):(t-1)]
    Xtrain <- newX[trainPoints,]
    Ytrain <- newY[trainPoints]
    XCal <- newX[calpoints,]
    YCal <- newY[calpoints]
    
    ### Fit regression algorithms on training setting
    mfit1 <- lm(Ytrain ~ Xtrain)
    mfit2 <- glmnet(Xtrain, Ytrain, lambda = lambda, standardize = FALSE, intercept = TRUE, alpha = 0)
    mfit3 <- glmnet(Xtrain, Ytrain, lambda = lambda, standardize = FALSE, intercept = TRUE, alpha = 1)
    
    ### Compute conformity score on calibration set and on new data example for model 1
    predForCal1 <- cbind(rep(1,nrow(XCal)),XCal)%*%mfit1$coef
    scores1 <- abs(predForCal1 - YCal)
    predt1 <- as.numeric(c(1, X[t,])%*%mfit1$coef)
    
    ### Compute conformity score on calibration set and on new data example for model 2
    beta2 <- as.vector(coef(mfit2))
    predForCal2 <- cbind(rep(1,nrow(XCal)),XCal)%*%beta2
    scores2 <- abs(predForCal2 - YCal)
    predt2 <- as.numeric((c(1, X[t,]))%*%beta2)
    
    ### Compute conformity score on calibration set and on new data example for model 3
    beta3 <- as.vector(coef(mfit3))
    predForCal3 <- cbind(rep(1,nrow(XCal)),XCal)%*%beta3
    scores3 <- abs(predForCal3 - YCal)
    predt3 <- as.numeric((c(1, X[t,]))%*%beta3)
    
    ### Compute errt for both methods
    confQuantnoAdapt1 <- quantile(scores1,probs=1-alpha)
    confQuantnoAdapt2 <- quantile(scores2,probs=1-alpha)
    confQuantnoAdapt3 <- quantile(scores3,probs=1-alpha)
    confQuantsnoAdapt  <- matrix(NA, nrow = 3, ncol = 2)
    confQuantsnoAdapt[1,] <- c(predt1 - confQuantnoAdapt1, predt1 + confQuantnoAdapt1)
    confQuantsnoAdapt[2,] <- c(predt2 - confQuantnoAdapt2, predt2 + confQuantnoAdapt2)
    confQuantsnoAdapt[3,] <- c(predt3 - confQuantnoAdapt3, predt3 + confQuantnoAdapt3)
    
    piNoAdapt <- majority_vote(confQuantsnoAdapt, rep(1,3), 0.5)
    for(j in nrow(piNoAdapt)){
      noAdaptErrorSeq[t-tinit+1] <- 1 - noAdaptErrorSeq[t-tinit+1]*(as.numeric(piNoAdapt[j,1] < Y[t] && Y[t] < piNoAdapt[j,2]))
    }

    if(alphat >=1){
      adaptErrSeq[t-tinit+1] <- 1
      confQuantAdapt <- 0
      #piAdapt[t-tinit+1,] <- c(predt1 - confQuantAdapt, predt1 + confQuantAdapt)
      wTrajectory[t-tinit+1,] <- wt/sum(wt)
      losst <- rep(0, 3)
      lossIter[t-tinit+1,] <- losst
    }else if (alphat <=0){
      adaptErrSeq[t-tinit+1] <- 0
      confQuantAdapt <- Inf
      #piAdapt[t-tinit+1,] <- c(predt1 - confQuantsnoAdapt, predt1 + confQuantsnoAdapt)
      wTrajectory[t-tinit+1,] <- wt/sum(wt)
      losst <- rep(1, 3)
      lossIter[t-tinit+1,] <- losst
    }else{
      confQuantAdapt1 <- quantile(scores1,probs=1-alphat)
      confQuantAdapt2 <- quantile(scores2,probs=1-alphat)
      confQuantAdapt3 <- quantile(scores3,probs=1-alphat)
      confQuants  <- matrix(NA, nrow = 3, ncol = 2)
      confQuants[1,] <- c(predt1 - confQuantAdapt1, predt1 + confQuantAdapt1)
      confQuants[2,] <- c(predt2 - confQuantAdapt2, predt2 + confQuantAdapt2)
      confQuants[3,] <- c(predt3 - confQuantAdapt3, predt3 + confQuantAdapt3)
      piAdapt <- majority_vote(confQuants, wt, 0.5)
      for(j in nrow(piNoAdapt)){
        adaptErrSeq[t-tinit+1] <- 1-adaptErrSeq[t-tinit+1]*(as.numeric(piAdapt[j,1] < Y[t] && Y[t] < piAdapt[j,2]))
      }
      ## update weights
      losst <- pgamma(confQuants[,2]-confQuants[,1], 0.1, 0.1)
      lossIter[t-tinit+1,] <- losst
      wTrajectory[t-tinit+1,] <- wt/sum(wt)
      wt <- adahedge(matrix(lossIter[1:(t-tinit+1),], t-tinit+1, 3))$weights[t-tinit+1,]
    }
    
    ## update alphat
    alphaTrajectory[t-tinit+1] <- alphat
    alphat <- alphat + gamma*(alpha-adaptErrSeq[t-tinit+1])
    
    if(t %% 100 == 0){
      print(sprintf("Done %i time steps",t))
    }
  }
  return(list(alpha_t=alphaTrajectory,
              AdaptErr=adaptErrSeq,
              noAdaptErr=noAdaptErrorSeq,
              piAdapt=piAdapt,
              piNoAdapt=piNoAdapt,
              weights=wTrajectory))
}

rMixAdap <- runMixAdapt(elec$transfer, as.matrix(elec[,1:4]), alpha = 0.05, gamma = 0.005)

data.plot <- data.frame(
  iter = 1:N,
  etaF = 1- stats::filter(rMix$AdaptErr, rep(1/400, 400)),
  AdaptEta = 1- stats::filter(rMix$noAdaptErr, rep(1/400, 400)),
  bern = stats::filter(rbinom(N, 1, 0.95), rep(1/400, 400))
)

pmA <- ggplot(data.plot, aes(x = iter)) +
  geom_line(aes(y = etaF, color = "Adapt"), linetype = "solid") +
  geom_line(aes(y = AdaptEta, color = "No Adapt"), linetype = "solid") +
  geom_line(aes(y = bern, color = "Bern 0.95"), linetype = "solid") +
  geom_hline(yintercept = 0.95, color = "black", linetype = "dashed") +
  labs(title = "", x = "Iter", y = "Local Level Coverage", color = "") +
  scale_color_manual(values = c("Adapt" = "blue", "No Adapt" = "red", "Bern 0.95" = "gray")) +
  theme_minimal() + theme(legend.position = "bottom")+
  ylim(0.8,1)
pmA

data.plot <- data.frame(
  iter = 1:N,
  etaF = 1 - stats::filter(rMixAdap$AdaptErr, rep(1/400, 400)),
  AdaptEta = 1- stats::filter(rMixAdap$noAdaptErr, rep(1/400, 400)),
  bern = stats::filter(rbinom(N, 1, 0.95), rep(1/400, 400))
)
pmB <- ggplot(data.plot, aes(x = iter)) +
  geom_line(aes(y = etaF, color = "Adapt"), linetype = "solid") +
  geom_line(aes(y = AdaptEta, color = "No Adapt"), linetype = "solid") +
  geom_line(aes(y = bern, color = "Bern 0.95"), linetype = "solid") +
  geom_hline(yintercept = 0.95, color = "black", linetype = "dashed") +
  labs(title = "", x = "Iter", y = "Local Level Coverage", color = "") +
  scale_color_manual(values = c("Adapt" = "blue", "No Adapt" = "red", "Bern 0.95" = "gray")) +
  theme_minimal() + theme(legend.position = "bottom")+
  ylim(0.8,1)
pmB

grid.arrange(pmA, pmB, ncol = 2)
