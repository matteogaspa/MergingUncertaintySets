rm(list = ls())
library(glmnet)
library(ggplot2)
library(dplyr)
library(gridExtra)
library(mvtnorm)
source("utils.R")
set.seed(12345)

# Create data -----
ind_1 <- c(1:50, 101:150, 251:350 ,451:550, 651:750, 851:950, 1051:1150, 1251:1350)
ind_2 <- c(51:100, 151:250, 351:450, 551:650, 751:850, 951:1050, 1151:1250, 1351:1450)
N <- length(ind_1) + length(ind_2)
Sigma <- matrix(c(1, .0, .0, 1), 2, 2)
X <- rmvnorm(N, sigma = Sigma)
x1 <- X[,1]
x2 <- X[,2]
beta <- 2
y <- rep(NA, N)
y[ind_1] <- beta*x1[ind_1] + rnorm(length(ind_1))
y[ind_2] <- beta*x2[ind_2] + rnorm(length(ind_2))


# lm -----
runlm <- function(Y,X,alpha,gamma,tinit = 100,splitSize = 0.5,updateMethod = "Simple",momentumBW=0.95){
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
    trainPoints <- sample(1:(tinit-1),round(splitSize*(tinit-1)))
    calpoints <- (1:(tinit-1))[-trainPoints]
    newX <- X[(t-tinit+1):(t-1)]
    newY <- Y[(t-tinit+1):(t-1)]
    Xtrain <- newX[trainPoints]
    Ytrain <- newY[trainPoints]
    XCal <- newX[calpoints]
    YCal <- newY[calpoints]
    
    ### Fit regression on training setting
    lmfit <- lm(Ytrain ~ Xtrain)
    
    ### Compute conformity score on calibration set and on new data example
    predForCal <- cbind(rep(1,NROW(XCal)),XCal)%*%lmfit$coef
    scores <- abs(predForCal - YCal)
    predt <- as.numeric(c(1, X[t])%*%lmfit$coef)
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
  }
  return(list(alpha_t=alphaTrajectory,
              AdaptErr=adaptErrSeq,
              noAdaptErr=noAdaptErrorSeq,
              piAdapt=piAdapt,
              piNoAdapt=piNoAdapt))
}

lm1 <- runlm(Y=y, X=x1, alpha = 0.05, gamma = 0.005)

set.seed(12345)
data.plot <- data.frame(
  iter = 1:NROW(lm1$AdaptErr),
  Adapt = 1 - lm1$AdaptErr %>% stats::filter(., rep(1/100, 100)),
  noAdapt = 1 - lm1$noAdaptErr %>% stats::filter(., rep(1/100, 100)),
  Bernoulli = rbinom(NROW(lm1$AdaptErr), 1, 0.95) %>% stats::filter(., rep(1/100, 100))
)



lm2 <- runlm(y, x2, alpha = 0.05, gamma = 0.005)
set.seed(12345)
data.plot <- data.frame(
  iter = 1:NROW(lm2$AdaptErr),
  Adapt = 1 - lm2$AdaptErr %>% stats::filter(., rep(1/100, 100)),
  noAdapt = 1 - lm2$noAdaptErr %>% stats::filter(., rep(1/100, 100)),
  Bernoulli = rbinom(NROW(lm2$AdaptErr), 1, 0.95) %>% stats::filter(., rep(1/100, 100))
)


# merging plot -----
loss <- cbind(lm1$piAdapt[,2] - lm1$piAdapt[,1], 
              lm2$piAdapt[,2] - lm2$piAdapt[,1])
loss.matrix <- pgamma(loss, 1, 0.1)
hedAlg <- adahedge(loss.matrix)

par(mfrow = c(1,2))
plot(cumsum(loss.matrix[,1]), type ="l", ylab = "cumulative loss")
lines(cumsum(loss.matrix[,2]), type = "l", col = 2)
w_12 <- which(cumsum(loss.matrix[,2]) > cumsum(loss.matrix[,1]))
points(w_12, rep(0, length(w_12)), col = "green", pch = "_")
legend("topleft", c("l1","l2","cumsum(l2) > cumsum(l1)"), col=1:3, lwd = 1, cex = 0.5)

plot(hedAlg$weights[,1], type = "l", ylim = c(-0.2, 1.2), ylab = "weights", xlab = "Iter")
lines(hedAlg$weights[,2], col = 2)
points(w_12, rep(-0.2, length(w_12)), col = "green", pch = "_")
legend("topleft", c(expression(w[1]), expression(w[2])), col = 1:2, lwd = 2, cex = 0.5)

plot(hedAlg$eta, type = "l", ylab = "eta")


# simulation -----
n_iter <- 2000
weights_B1 <- weights_B2 <- cumsum_lm1 <- cumsum_lm2 <- matrix(NA, nrow = N-99, ncol = n_iter)
lcov_1 <- lcov_2 <- matrix(NA, nrow = N-99, ncol = n_iter)
for(i in 1:n_iter){
  # generate data
  X <- rmvnorm(N, sigma = Sigma)
  x1 <- X[,1]
  x2 <- X[,2]
  beta <- 2
  y <- rep(NA, N)
  y[ind_1] <- beta*x1[ind_1] + rnorm(length(ind_1))
  y[ind_2] <- beta*x2[ind_2] + rnorm(length(ind_2))
  
  lm1 <- runlm(y, x1, alpha = 0.05, gamma = 0.005)
  lm2 <- runlm(y, x2, alpha = 0.05, gamma = 0.005)
  loss.matrix <- cbind(lm1$piAdapt[,2] - lm1$piAdapt[,1], 
                       lm2$piAdapt[,2] - lm2$piAdapt[,1])
  loss.matrix <- pgamma(loss.matrix, 1, 0.1)
  cumsum_lm1[,i] <- cumsum(loss.matrix[,1])
  cumsum_lm2[,i] <- cumsum(loss.matrix[,2])
  hedAlg <- adahedge(loss.matrix)
  weights_B1[,i] <- hedAlg$weights[,1]
  weights_B2[,i] <- hedAlg$weights[,2]
  lcov_1[,i] <- lm1$AdaptErr
  lcov_2[,i] <- lm2$AdaptErr
  if(i %% 100 == 0){
    print(sprintf("Done %i time steps",i))
  }
}


data.plot1 <- data.frame(
  t = 1:(N-99),
  w1 = rowMeans(weights_B1),
  w2 = rowMeans(weights_B2)
)

data.plot2 <- data.frame(
  t = 1:(N-99),
  L1 = rowMeans(cumsum_lm1),
  L2 = rowMeans(cumsum_lm2),
  a1 = I((rowMeans(cumsum_lm1) - rowMeans(cumsum_lm2))>0)
)



pl1<-ggplot(data.plot2, aes(x = t)) +
  geom_line(aes(y = L1, color = "L1"), size = 1) +
  geom_line(aes(y = L2, color = "L2"), size = 1) +
  labs(title = "",
       x = "t",
       y = "L") +
  scale_color_manual(values = c("L1" = "tan2", "L2" = "seagreen")) +
  geom_segment(data = subset(data.plot2, a1), aes(x = t, xend = t, y = 0, yend = 800), color = "gray", alpha = 0.02) +
  theme_classic() +
  theme(legend.position = "bottom") +
  guides(color = guide_legend(title = "")) 

pl2<-ggplot(data.plot1, aes(x = t)) +
  geom_line(aes(y = w1, color = "w1"), size = 1) +
  geom_line(aes(y = w2, color = "w2"), size = 1) +
  labs(title = "",
       x = "t",
       y = "Weights") +
  scale_color_manual(values = c("w1" = "tan2", "w2" = "seagreen")) +
  theme_classic() +
  theme(legend.position = "bottom") +
  geom_segment(data = subset(data.plot2, a1), aes(x = t, xend = t, y = 0, yend = 1), color = "gray", alpha = 0.02) +
  guides(color = guide_legend(title = ""))
grid.arrange(pl1, pl2, ncol = 2)

1-mean(lcov_1); 1-mean(lcov_2)
