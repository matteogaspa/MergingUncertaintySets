# MoM estimator
rm(list = ls())
library(sn)
library(ggplot2)
source("utils.R")

# Functions -----
MoM_fn <- function(x, B){
  # x: data
  # B: number of splits
  n <- length(x)
  x <- x[sample(1:n, n, replace = F)]
  m <- floor(n / B)
  sub_x <- split(x, rep(1:B, each = m, length.out = n))
  means <- as.vector(unlist(lapply(sub_x, mean)))
  return(median(means))
}

MoMoM_fn <- function(x, B, K){
  # x: data
  # B: number of splits
  # K: number of rep
  res <- rep(NA, K)
  for(i in 1:K){
    res[i] <- MoM_fn(x, B)
  }
  return(median(res))
}

HulC_fn <- function(x, B1, B2){
  # x: data
  # B1: number of splits
  # B2: number of splits (for the MoM)
  n <- length(x)
  x <- x[sample(1:n, n, replace = F)]
  m <- floor(n / B1)
  sub_x <- split(x, rep(1:B1, each = m, length.out = n))
  # insert the estimator
  thetahat <- as.vector(unlist(lapply(sub_x, function(x) MoM_fn(x, B2))))
  return(c(min(thetahat), max(thetahat)))
}



# Example 1 -----
set.seed(5)
n   <- 210  # number of obs
K   <- 50   # number of reps
B2  <- 7    # number of splits for the MoM
cis <- matrix(NA, nrow = K, ncol = 2)
cisM<- matrix(NA, nrow = K, ncol = 2)
cisE<- matrix(NA, nrow = K, ncol = 2)
# simulate data
x <- rt(n, df = 3)

for(k in 1:K){
  B1      <- sample(c(5,6), 1, prob = c(0.6, 0.4))
  cis[k,] <- HulC_fn(x, B1 = B1, B2 = B2)
  if(k==1){
    cisM[k,]<- cis[1,]
    cisE[k,]<- cis[1,]
  }else{
    cisM[k,]<- majority_vote(cis[1:k,], rep(1/k, k))
    cisE[k,]<- majority_vote(rbind(cisE[k-1,], cisM[k,]), rep(1/2, 2))
  }
}



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
  theme_minimal() + theme(legend.position = "none", legend.title = element_blank()) + ylim(min(lo), max(up)) +
  theme(legend.text = element_text(size = 12),
        title = element_text(size = 15), axis.title = element_text(size = 15), legend.title = element_blank())
plot1a

plot1b <- ggplot(data.plot, aes(x = k)) +
  geom_point(aes(y = lo, color = "Majority Vote")) +
  geom_point(aes(y = up, color = "Majority Vote")) +
  geom_segment(aes(x = k, xend = k, y = lo, yend = up), color = "black") +
  labs(title = "Confidence sets",
       x = "# of replications (K)",
       y = "Confidence Intervals") +
  scale_color_manual(values = c("Majority Vote" = "black", "Exch. Majority Vote" = "red")) +
  theme_minimal() + theme(legend.position = "none", legend.title = element_blank()) + ylim(min(lo), max(up)) +
  theme(legend.text = element_text(size = 12),
        title = element_text(size = 15), axis.title = element_text(size = 15), legend.title = element_blank())
plot1b

pl1 <- gridExtra::grid.arrange(plot1b, plot1a, ncol = 2)

## simulation -----
n   <- 210  # number of obs
K   <- 100  # number of reps
B2  <- 7    # number of splits for the MoM
L   <- 5000 # number of sims
len_resM <- cov_resM <- matrix(NA, nrow = L, ncol = K)
len_resE <- cov_resE <- matrix(NA, nrow = L, ncol = K)
len_resS <- cov_resS <- matrix(NA, nrow = L, ncol = K)


set.seed(5)
for(i in 1:L){
  cis <- matrix(NA, nrow = K, ncol = 2)
  cisM<- vector("list", K)
  cisE<- vector("list", K)
  # simulate data
  x <- rt(n, df = 3)
  for(k in 1:K){
    B1       <- sample(5:6, 1, prob = c(0.6, 0.4))
    cis[k,]  <- HulC_fn(x, B1 = B1, B2 = B2)
    
    if(k==1){
      cisM[[k]]<- cis[1,]
      cisE[[k]]<- cis[1,]
    }else{
      cisM[[k]]<- majority_vote(cis[1:k,], rep(1/k, k))
      newM     <- rbind(cisE[[k-1]], cisM[[k]])
      cisE[[k]]<- majority_vote(newM, rep(1/NROW(newM), NROW(newM)))
    }
    len_resM[i,k] <- len_ints(cisM[[k]])
    len_resE[i,k] <- len_ints(cisE[[k]])
    len_resS[i,k] <- cis[k,2]-cis[k,1]
    cov_resM[i,k] <- cov_ints(cisM[[k]], 0)
    cov_resE[i,k] <- cov_ints(cisE[[k]], 0)
    cov_resS[i,k] <- I(cis[k,1] <= 0 & 0 <= cis[k,2])
  }
  if(i %% 10 == 0) cat(i, "\n")
}

colMeans(cov_resM); colMeans(cov_resE); mean(cov_resS)
colMeans(len_resM); colMeans(len_resE); mean(len_resS)

# Plots -----
data.plot1 <- data.frame(
  K = 1:K,
  mv = colMeans(cov_resM),
  emv = colMeans(cov_resE),
  sic = colMeans(cov_resS)
)

p1 <- ggplot(data.plot1, aes(x = K)) +
  geom_line(aes(y = mv, color = "Maj. Vote"), linetype = "dashed") +
  geom_line(aes(y = emv, color = "Exch. Maj. Vote"), linetype = "longdash") +
  geom_line(aes(y = sic, color = "Single Ints")) +
  geom_hline(yintercept = 0.9, linetype = "dashed", col = "gray") +
  geom_hline(yintercept = 0.95, linetype = "dashed", col = "gray") +
  labs(title = "",
       x = "# of replications (K)",
       y = "Coverage") +
  theme_minimal() +
  scale_color_manual(values = c("Maj. Vote" = "blue", "Exch. Maj. Vote" = "red", "Single Ints" = "forestgreen")) +
  theme(legend.position = "bottom", legend.text = element_text(size = 12),
        title = element_text(size = 15), axis.title = element_text(size = 15), legend.title = element_blank())

data.plot2 <- data.frame(
  K = 1:K,
  mv = colMeans(len_resM),
  emv = colMeans(len_resE),
  sic = colMeans(len_resS)
)

p2 <- ggplot(data.plot2, aes(x = K)) +
  geom_line(aes(y = mv, color = "Maj. Vote"), linetype = "dashed") +
  geom_line(aes(y = emv, color = "Exch. Maj. Vote"), linetype = "longdash") +
  geom_line(aes(y = sic, color = "Single Ints")) +
  labs(title = "",
       x = "# of replications (K)",
       y = "Length") +
  theme_minimal() +
  scale_color_manual(values = c("Maj. Vote" = "blue", "Exch. Maj. Vote" = "red", "Single Ints" = "forestgreen")) +
  theme(legend.position = "bottom", legend.text = element_text(size = 12),
        title = element_text(size = 15), axis.title = element_text(size = 15), legend.title = element_blank())
p1;p2

pl2 <- gridExtra::grid.arrange(p1, p2, ncol = 2)
pl3 <- gridExtra::grid.arrange(pl1, pl2, nrow = 1)

