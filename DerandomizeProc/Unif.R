rm(list = ls())
library(sn)
library(ggplot2)
library(patchwork)
source("majorityVote.R")

# FUNCTIONS -----
thetahat_fn <- function(x){
  # x: data
  x <- sort(x, decreasing = T)
  2*x[1] - x[2]
}
est <- 1:1000
for(i in 1:1000){
  est[i] <- thetahat_fn(runif(5))
}

HulC_fn <- function(x, alpha=.05){
  # x: data
  # alpha: confidence level
  n <- length(x)
  x <- x[sample(1:n, n, replace = F)]
  B <- ceiling(1 - log(alpha,2))
  pb0<- 1/2^(B-1)
  pb1<- 1/2^(B-2)
  tau<- (alpha-pb0)/(pb1-pb0)
  Bs <- sample(c(B,B-1),1,F,c(tau,1-tau))
  m <- floor(n / Bs)
  sub_x <- split(x, rep(1:Bs, each = m, length.out = n))
  # insert the estimator
  thetahat <- as.vector(unlist(lapply(sub_x, function(x) thetahat_fn(x))))
  return(c(min(thetahat), max(thetahat)))
}

# AN EXAMPLE -----
set.seed(1234)
n   <- 60   # number of obs
K   <- 50   # number of reps
cis <- matrix(NA, nrow = K, ncol = 2)
cisM<- matrix(NA, nrow = K, ncol = 2)
cisE<- matrix(NA, nrow = K, ncol = 2)

# simulate data
x <- runif(n, 0, 2)

for(k in 1:K){
  cis[k,] <- HulC_fn(x)
  if(k==1){
    cisM[k,]<- cis[1,]
    cisE[k,]<- cis[1,]
  }else{
    cisM[k,]<- majority_vote(cis[1:k,], rep(1/k, k))
    cisE[k,]<- majority_vote(rbind(cisE[k-1,], cisM[k,]), rep(1/2, 2))
  }
}
plot(cisM[,1], pch = 20, ylim = c(0, 4))
points(cisM[,2], pch = 20)
points(cisE[,1], col = 2, pch = "x")
points(cisE[,2], col = 2, pch = "x")  


data.plot <- data.frame(
  "k" <- 1:K,
  "loM" <- cisM[,1],
  "upM" <- cisM[,2],
  "loE" <- cisE[,1],
  "upE" <- cisE[,2],
  "lo"  <- cis[,1],
  "up"  <- cis[,2]
)

plot1a <- ggplot(data.plot, aes(x = k)) +
  geom_point(aes(y = loM, color = "Majority Vote")) +
  geom_point(aes(y = upM, color = "Majority Vote")) +
  geom_segment(aes(x = k, xend = k, y = loM, yend = upM), color = "black") +
  labs(title = "Majority vote confidence sets",
       x = "# of replications (K)",
       y = "Confidence Intervals") +
  scale_color_manual(values = c("Majority Vote" = "black", "Exch. Majority Vote" = "red")) +
  theme_minimal() + theme(legend.position = "none", legend.title = element_blank()) + ylim(min(lo), max(up))
plot1a

plot1b <- ggplot(data.plot, aes(x = k)) +
  geom_point(aes(y = lo, color = "Majority Vote")) +
  geom_point(aes(y = up, color = "Majority Vote")) +
  geom_segment(aes(x = k, xend = k, y = lo, yend = up), color = "black") +
  labs(title = "Confidence intervals",
       x = "# of replications (K)",
       y = "Confidence Intervals") +
  scale_color_manual(values = c("Majority Vote" = "black", "Exch. Majority Vote" = "red")) +
  theme_minimal() + theme(legend.position = "none", legend.title = element_blank()) + ylim(min(lo), max(up))
plot1b

pl1 <- gridExtra::grid.arrange(plot1b, plot1a, ncol = 2)

## simulation -----
n   <- 60   # number of obs
K   <- 50   # number of reps
L   <- 5000 # number of sims
len_resM <- cov_resM <- matrix(NA, nrow = L, ncol = K)
len_resE <- cov_resE <- matrix(NA, nrow = L, ncol = K)
len_resS <- cov_resS <- matrix(NA, nrow = L, ncol = K)


set.seed(123)
for(i in 1:L){
  cis <- matrix(NA, nrow = K, ncol = 2)
  cisM<- matrix(NA, nrow = K, ncol = 2)
  cisE<- matrix(NA, nrow = K, ncol = 2)
  # simulate data
  x <- runif(n, 0, 2)
  for(k in 1:K){
    cis[k,]  <- HulC_fn(x)
    
    if(k==1){
      cisM[k,]<- cis[1,]
      cisE[k,]<- cis[1,]
    } else {
      mv      <- majority_vote(cis[1:k,], rep(1/k, k))
      if(is.na(mv[1])){ mv <- matrix(c(0,0), ncol=2)}
      cisM[k,]<- c(min(mv[,1]), max(mv[,2]))
      mve     <- majority_vote(rbind(cisE[k-1,], cisM[k,]), rep(1/2, 2))
      if(is.na(mve[1])){ mve <- matrix(c(0,0), ncol=2)}
      cisE[k,]<- c(min(mve[,1]), max(mve[,2]))
    }
    len_resM[i,k] <- cisM[k,2]-cisM[k,1]
    len_resE[i,k] <- cisE[k,2]-cisE[k,1]
    len_resS[i,k] <- cis[k,2]-cis[k,1]
    cov_resM[i,k] <- I(cisM[k,1] <= 2 & 2 <= cisM[k,2])
    cov_resE[i,k] <- I(cisE[k,1] <= 2 & 2 <= cisE[k,2])
    cov_resS[i,k] <- I(cis[k,1] <= 2 & 2 <= cis[k,2])
  }
  if(i %% 100 == 0) cat(i, "\n")
}

colMeans(cov_resM); colMeans(cov_resE); mean(cov_resS)
colMeans(len_resM); colMeans(len_resE); mean(len_resS)

# plots
data.plot1 <- data.frame(
  K = 1:K,
  mv = colMeans(cov_resM),
  emv = colMeans(cov_resE),
  sic = colMeans(cov_resS)
)

p1 <- ggplot(data.plot1, aes(x = K)) +
  geom_line(aes(y = mv, color = "MV"), linetype = "solid") +
  geom_line(aes(y = emv, color = "Exch. MV"), linetype = "longdash") +
  geom_line(aes(y = sic, color = "Single Ints")) +
  geom_hline(yintercept = 0.9, linetype = "dashed", col = "gray") +
  geom_hline(yintercept = 0.95, linetype = "dashed", col = "gray") +
  labs(title = "",
       x = "# of replications (K)",
       y = "Coverage") +
  theme_minimal() +
  scale_color_manual(values = c("MV" = "blue", "Exch. MV" = "red", "Single Ints" = "forestgreen")) +
  theme(legend.position = "bottom", legend.text = element_text(size = 10),
        title = element_text(size = 12), axis.title = element_text(size = 12), legend.title = element_blank())

data.plot2 <- data.frame(
  K = 1:K,
  mv = colMeans(len_resM),
  emv = colMeans(len_resE),
  sic = colMeans(len_resS)
)

p2 <- ggplot(data.plot2, aes(x = K)) +
  geom_line(aes(y = mv, color = "MV"), linetype = "solid") +
  geom_line(aes(y = emv, color = "Exch. MV"), linetype = "longdash") +
  geom_line(aes(y = sic, color = "Single Ints")) +
  labs(title = "",
       x = "# of replications (K)",
       y = "Length") +
  theme_minimal() +
  scale_color_manual(values = c("MV" = "blue", "Exch. MV" = "red", "Single Ints" = "forestgreen")) +
  theme(legend.position = "bottom", legend.text = element_text(size = 10),
        title = element_text(size = 12), axis.title = element_text(size = 12), legend.title = element_blank())


pl2 <- gridExtra::grid.arrange(p1, p2, ncol = 2)
pl3 <- gridExtra::grid.arrange(pl1, pl2, nrow = 1)


