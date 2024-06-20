rm(list = ls())
library(dplyr)
library(DoubleML)
library(mlr3)
library(mlr3learners)
library(causaldata)
source("majorityVote.R")

# Fix seed 
set.seed(1)

# Load Data
data("nsw_mixtape")
nsw_mixtape <- as.data.frame(nsw_mixtape)
summary(nsw_mixtape)
head(nsw_mixtape)

# remove ID
nsw_mixtape <- nsw_mixtape[,-1]
nsw_mixtape

# change in log 
nsw_mixtape$re74 <- log(nsw_mixtape$re74 + 1)
nsw_mixtape$re75 <- log(nsw_mixtape$re75 + 1)
nsw_mixtape$re78 <- log(nsw_mixtape$re78 + 1)

# choosing regression and classification algorithms
ml_g = lrn("regr.ranger",
           num.trees = 40, mtry = 2,
           min.node.size = 2, max.depth = 5)
ml_m = lrn("classif.ranger",
           num.trees = 40, mtry = 2,
           min.node.size = 2, max.depth = 5)

obj_dml_data <- DoubleMLData$new(nsw_mixtape, y_col="re78", d_cols="treat")
dml_irm_obj <- DoubleMLIRM$new(obj_dml_data, ml_g, ml_m)
dml_irm_obj$fit()
dml_irm_obj$summary()

# multiple intervals
alpha <- 0.05              # coverage level
quan_a <- qnorm(1-alpha/2) # quantile
K <- 50                    # number of replications

cis <- matrix(NA, nrow = K, 2) 
cisM<- matrix(NA, nrow = K, ncol = 2)
cisE<- matrix(NA, nrow = K, ncol = 2)

set.seed(1)
for(k in 1:K){
  dml_irm_obj$fit()
  dml_irm_obj$summary()
  cis[k,] <-  c(dml_irm_obj$summary()[1] - quan_a*dml_irm_obj$summary()[2],dml_irm_obj$summary()[1] + quan_a*dml_irm_obj$summary()[2]) 
  if(k==1){
    cisM[k,]<- cis[1,]
    cisE[k,]<- cis[1,]
  }else{
    cisM[k,]<- majority_vote(cis[1:k,], rep(1/k, k))
    cisE[k,]<- majority_vote(rbind(cisE[k-1,], cisM[k,]), rep(1/2, 2))
  }
}

data.plot <- data.frame(
  "k" = 1:K,
  "loM" = cisM[,1],
  "upM" = cisM[,2],
  "loE" = cisE[,1],
  "upE" = cisE[,2]
)

plot1a <- ggplot(data = data.plot, aes(x = k)) +
  geom_point(aes(y = loM, color = "Majority Vote")) +
  geom_point(aes(y = upM, color = "Majority Vote")) +
  geom_segment(aes(x = k, xend = k, y = loM, yend = upM), color = "black") +
  labs(title = "Cross-fitting estimator",
       x = "# of replications (K)",
       y = "Confidence Intervals") +
  scale_color_manual(values = c("Majority Vote" = "black", "Exch. Majority Vote" = "red")) +
  theme_minimal() + theme(legend.position = "none", legend.title = element_blank()) + ylim(c(0, 2))
plot1a


cis <- matrix(NA, nrow = K, 2) 
cisM<- matrix(NA, nrow = K, ncol = 2)
cisE<- matrix(NA, nrow = K, ncol = 2)

set.seed(2)
for(k in 1:K){
  dml_irm_obj$fit()
  dml_irm_obj$summary()
  cis[k,] <-  c(dml_irm_obj$summary()[1] - quan_a*dml_irm_obj$summary()[2],dml_irm_obj$summary()[1] + quan_a*dml_irm_obj$summary()[2]) 
  if(k==1){
    cisM[k,]<- cis[1,]
    cisE[k,]<- cis[1,]
  }else{
    cisM[k,]<- majority_vote(cis[1:k,], rep(1/k, k))
    cisE[k,]<- majority_vote(rbind(cisE[k-1,], cisM[k,]), rep(1/2, 2))
  }
}

data.plot2 <- data.frame(
  "k" = 1:K,
  "loM" = cisM[,1],
  "upM" = cisM[,2],
  "loE" = cisE[,1],
  "upE" = cisE[,2]
)

plot1b <- ggplot(data=data.plot2, aes(x = k)) +
  geom_point(aes(y = loM, color = "Majority Vote")) +
  geom_point(aes(y = upM, color = "Majority Vote")) +
  geom_segment(aes(x = k, xend = k, y = loM, yend = upM), color = "black") +
  labs(title = "Cross-fitting estimator",
       x = "# of replications (K)",
       y = "Confidence Intervals") +
  scale_color_manual(values = c("Majority Vote" = "black", "Exch. Majority Vote" = "red")) +
  theme_minimal() + theme(legend.position = "none", legend.title = element_blank()) + ylim(c(0, 2))
plot1b

gridExtra::grid.arrange(plot1a, plot1b, ncol = 2)


