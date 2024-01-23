rm(list = ls())
library(dplyr)
library(glmnet)
library(randomForest)
library(polspline)
library(e1071)
library(ggplot2)
library(mlr)
library(gridExtra)
library(mvtnorm)
source("utils_simul_rcps.R")

# Generate data -----
nclass <- 10
npoint <- 600
dati   <- NULL
set.seed(1234)
for(i in 1:nclass){
  cors <- runif(1, -1, 1)
  sds <- runif(1, 3, 5)
  pclass <- cbind(rep(i, npoint), rmvnorm(npoint, mean = c(i,i), sigma = matrix(c(sds^2, sds*cors, sds*cors, sds^2), 2, 2)), rnorm(npoint), rnorm(npoint))
  dati <- rbind(dati, pclass)
}
plot(dati[,2:3], col = dati[,1], pch = 20)

colnames(dati) <- c("y", "x1", "x2", "x3", "x4")
dati <- as.data.frame(dati)
dati$y <- as.factor(dati$y)

# Split in train, calibration and test ----
simul_task <- makeClassifTask(data = dati, target = "y")
indc <- rep(1:3, 2000)

# Train the models -----
## Neural Net -----
lrn.nn <- makeLearner("classif.nnet", predict.type = "prob")

m1 <- mlr::train(lrn.nn, simul_task, subset = which(indc == 1))
p1 <- predict(m1, task = simul_task, subset = which(indc == 2))$data[,3:12]

## Random Forest -----
lrn.rf <- makeLearner("classif.randomForest", predict.type = "prob")

m2 <-  mlr::train(lrn.rf, simul_task, subset = which(indc == 1))
p2 <- predict(m2, task = simul_task, subset = which(indc == 2))$data[,3:12]

## Xgboost -----
lrn.xg <- makeLearner("classif.gbm", predict.type = "prob")

m3 <- mlr::train(lrn.xg, simul_task, subset = which(indc == 1))
p3 <- predict(m3, task = simul_task, subset = which(indc == 2))$data[,3:12]

## CART ----
lrn.crt <- makeLearner("classif.rpart", predict.type = "prob")

m4 <- mlr::train(lrn.crt, simul_task, subset = which(indc == 1))
p4 <- predict(m4, task = simul_task, subset = which(indc == 2))$data[,3:12]

## QDA ----
lrn.qda <- makeLearner("classif.qda", predict.type = "prob")

m5 <- mlr::train(lrn.qda, simul_task, subset = which(indc == 1))
p5 <- predict(m5, task = simul_task, subset = which(indc == 2))$data[,3:12]

## MARS -----
lrn.mars <- makeLearner("classif.earth", predict.type = "prob")

m6 <- mlr::train(lrn.mars, simul_task, subset = which(indc == 1))
p6 <- predict(m6, task = simul_task, subset = which(indc == 2))$data[,3:12]

## LDA -----
lrn.lda <- makeLearner("classif.lda", predict.type = "prob")

m7 <- mlr::train(lrn.lda, simul_task, subset = which(indc == 1))
p7 <- predict(m5, task = simul_task, subset = which(indc == 2))$data[,3:12]

k <- 7 #number of algorithms

# Set lambda values -----
lambdas <- seq(0, 1, by = 0.01)
ycal <- dati$y[indc == 2]
M1 <- M2 <- M3 <- M4 <- M5 <- M6 <- M7 <- matrix(NA, nrow = nrow(p1), ncol = length(lambdas))
for(i in 1:length(lambdas)){
  M1[,i] <- get_loss_table(lambdas[i], p1, ycal)
  M2[,i] <- get_loss_table(lambdas[i], p2, ycal)
  M3[,i] <- get_loss_table(lambdas[i], p3, ycal)
  M4[,i] <- get_loss_table(lambdas[i], p4, ycal)
  M5[,i] <- get_loss_table(lambdas[i], p5, ycal)
  M6[,i] <- get_loss_table(lambdas[i], p6, ycal)
  M7[,i] <- get_loss_table(lambdas[i], p7, ycal)
}


l1 <- get_lhat(calib_loss_table = M1, lambdas, 0.05)
l2 <- get_lhat(calib_loss_table = M2, lambdas, 0.05)
l3 <- get_lhat(calib_loss_table = M3, lambdas, 0.05)
l4 <- get_lhat(calib_loss_table = M4, lambdas, 0.05)
l5 <- get_lhat(calib_loss_table = M5, lambdas, 0.05)
l6 <- get_lhat(calib_loss_table = M6, lambdas, 0.05)
l7 <- get_lhat(calib_loss_table = M7, lambdas, 0.05)

counts <- matrix(NA, nrow=nrow(p1), k)
counts[,1] <- apply(p1, 1, function(x) sum(ifelse(x>1-l1, 1, 0)))
counts[,2] <- apply(p2, 1, function(x) sum(ifelse(x>1-l2, 1, 0)))
counts[,3] <- apply(p3, 1, function(x) sum(ifelse(x>1-l3, 1, 0)))
counts[,4] <- apply(p4, 1, function(x) sum(ifelse(x>1-l4, 1, 0)))
counts[,5] <- apply(p5, 1, function(x) sum(ifelse(x>1-l5, 1, 0)))
counts[,6] <- apply(p6, 1, function(x) sum(ifelse(x>1-l6, 1, 0)))
counts[,7] <- apply(p7, 1, function(x) sum(ifelse(x>1-l7, 1, 0)))

counts <- as.data.frame(counts)
colnames(counts) <- c("NN", "RF", "XG_Boost", "CART", "QDA", "MARS", "LDA")

plot1<-ggplot(counts, aes(x = NN, fill = NN)) +
       geom_bar(stat="count", fill = "darkgoldenrod2") +
       ggtitle("NN") +
       xlab("Set size") +
       ylab("Frequency") +
       theme_minimal() 
plot2<-ggplot(counts, aes(x = RF, fill = RF)) +
       geom_bar(stat="count", fill = "dodgerblue1") +
       ggtitle("RF") +
       xlab("Set size") +
       ylab("Frequency") +
       theme_minimal()  
plot3<-ggplot(counts, aes(x = XG_Boost, fill = XG_Boost)) +
       geom_bar(stat="count", fill = "coral2") +
       ggtitle("XG Boost") +
       xlab("Set size") +
       ylab("Frequency") +
       theme_minimal()  
plot4<-ggplot(counts, aes(x = CART, fill = CART)) +
       geom_bar(stat="count", fill = "darkgreen") +
       ggtitle("CART") +
       xlab("Set size") +
       ylab("Frequency") +
       theme_minimal() 
plot5<-ggplot(counts, aes(x = QDA, fill = QDA)) +
       geom_bar(stat="count", fill = "darkorchid1") +
       ggtitle("QDA") +
       xlab("Set size") +
       ylab("Frequency") +
       theme_minimal() 
plot6<-ggplot(counts, aes(x = MARS, fill = MARS)) +
       geom_bar(stat="count", fill = "orange") +
       ggtitle("MARS") +
       xlab("Set size") +
       ylab("Frequency") +
       theme_minimal() 
plot7<-ggplot(counts, aes(x = LDA, fill = LDA)) +
       geom_bar(stat="count", fill = "lightblue") +
       ggtitle("LDA") +
       xlab("Set size") +
       ylab("Frequency") +
       theme_minimal() 
grid.arrange(plot1,plot2,plot3,plot4,plot5,plot6,plot7,nrow=1)


# Test points ----
pn1 <- predict(m1, task = simul_task, subset = which(indc == 3))$data[,3:12]
pn2 <- predict(m2, task = simul_task, subset = which(indc == 3))$data[,3:12]
pn3 <- predict(m3, task = simul_task, subset = which(indc == 3))$data[,3:12]
pn4 <- predict(m4, task = simul_task, subset = which(indc == 3))$data[,3:12]
pn5 <- predict(m5, task = simul_task, subset = which(indc == 3))$data[,3:12]
pn6 <- predict(m6, task = simul_task, subset = which(indc == 3))$data[,3:12]
pn7 <- predict(m7, task = simul_task, subset = which(indc == 3))$data[,3:12]

counts <- matrix(NA, nrow=nrow(pn1), k)
counts[,1] <- apply(pn1, 1, function(x) sum(ifelse(x>1-l1, 1, 0)))
counts[,2] <- apply(pn2, 1, function(x) sum(ifelse(x>1-l2, 1, 0)))
counts[,3] <- apply(pn3, 1, function(x) sum(ifelse(x>1-l3, 1, 0)))
counts[,4] <- apply(pn4, 1, function(x) sum(ifelse(x>1-l4, 1, 0)))
counts[,5] <- apply(pn5, 1, function(x) sum(ifelse(x>1-l5, 1, 0)))
counts[,6] <- apply(pn6, 1, function(x) sum(ifelse(x>1-l6, 1, 0)))
counts[,7] <- apply(pn7, 1, function(x) sum(ifelse(x>1-l7, 1, 0)))


counts <- as.data.frame(counts)
colnames(counts) <- c("NN", "RF", "XG_Boost", "CART", "QDA", "MARS", "LDA")

plot1<-ggplot(counts, aes(x = NN, fill = NN)) +
  geom_bar(stat="count", fill = "darkgoldenrod2") +
  ggtitle("NN") +
  xlab("Set size") +
  ylab("Frequency") +
  theme_minimal() + 
  xlim(0,10) +
  ylim(0,1000)  
plot2<-ggplot(counts, aes(x = RF, fill = RF)) +
  geom_bar(stat="count", fill = "dodgerblue1") +
  ggtitle("RF") +
  xlab("Set size") +
  ylab("Frequency") +
  theme_minimal()  + 
  xlim(0,10) +
  ylim(0,1000)
plot3<-ggplot(counts, aes(x = XG_Boost, fill = XG_Boost)) +
  geom_bar(stat="count", fill = "coral2") +
  ggtitle("XG Boost") +
  xlab("Set size") +
  ylab("Frequency") +
  theme_minimal()  + 
  xlim(0,10) +
  ylim(0,1000)
plot4<-ggplot(counts, aes(x = CART, fill = CART)) +
  geom_bar(stat="count", fill = "darkgreen") +
  ggtitle("CART") +
  xlab("Set size") +
  ylab("Frequency") +
  theme_minimal() + 
  xlim(0,10) +
  ylim(0,1000)
plot5<-ggplot(counts, aes(x = QDA, fill = QDA)) +
  geom_bar(stat="count", fill = "darkorchid1") +
  ggtitle("QDA") +
  xlab("Set size") +
  ylab("Frequency") +
  theme_minimal() + 
  xlim(0,10) +
  ylim(0,1000)
plot6<-ggplot(counts, aes(x = MARS, fill = MARS)) +
  geom_bar(stat="count", fill = "orange") +
  ggtitle("MARS") +
  xlab("Set size") +
  ylab("Frequency") +
  theme_minimal() + 
  xlim(0,10) +
  ylim(0,1000)
plot7<-ggplot(counts, aes(x = LDA, fill = LDA)) +
  geom_bar(stat="count", fill = "lightgreen") +
  ggtitle("LDA") +
  xlab("Set size") +
  ylab("Frequency") +
  theme_minimal() + 
  xlim(0,10) +
  ylim(0,1000)


grid.arrange(plot1,plot2,plot3,plot4,plot5,plot6,plot7,nrow=1)


# Obtain Majority vote sets ----
sets1 <- apply(pn1, 2, function(x) x>(1-l1))
sets2 <- apply(pn2, 2, function(x) x>(1-l2))
sets3 <- apply(pn3, 2, function(x) x>(1-l3))
sets4 <- apply(pn4, 2, function(x) x>(1-l4))
sets5 <- apply(pn5, 2, function(x) x>(1-l5))
sets6 <- apply(pn6, 2, function(x) x>(1-l6))
sets7 <- apply(pn7, 2, function(x) x>(1-l7))

ys <- 1:nclass
ytest <- dati$y[which(indc==3)]
maj_ints <- rand_maj_ints <- matrix(0, nrow = length(ytest), ncol = nclass)

for(i in 1:length(ytest)){
  vote <- cbind(loss_cvg(ys, sets1[i,]), loss_cvg(ys, sets2[i,]), loss_cvg(ys, sets3[i,]), loss_cvg(ys, sets4[i,]), 
                loss_cvg(ys, sets5[i,]), loss_cvg(ys, sets6[i,]), loss_cvg(ys, sets7[i,]))
  maj_ints[i,which(rowMeans(vote)<0.5)] <- 1
  rand_maj_ints[i,which(rowMeans(vote)<0.5-runif(1,0,0.5))] <- 1
}

counts_maj <- data.frame(maj_vote = rowSums(maj_ints),
                         rand_maj_vote = rowSums(rand_maj_ints))


plot8<-ggplot(counts_maj, aes(x = maj_vote, fill = maj_vote)) +
  geom_bar(stat="count", fill = "indianred2") +
  ggtitle("Majority Vote") +
  xlab("Set size") +
  ylab("Frequency") +
  theme_minimal() +
  xlim(0, 10) +
  ylim(0, 1000)
plot9<-ggplot(counts_maj, aes(x = rand_maj_vote, fill = rand_maj_vote)) +
  geom_bar(stat="count", fill = "royalblue3") +
  ggtitle("Randomized Majority Vote") +
  xlab("Set size") +
  ylab("Frequency") +
  theme_minimal() +
  xlim(0, 10) +
  ylim(0, 1000)

grid.arrange(plot1,plot2,plot3,plot4,plot5,plot6,plot7,plot8,plot9,nrow=3)


# Majority vote -----
loss_maj <- loss_rand_maj <- rep(NA, nrow(maj_ints))
for(i in 1:nrow(maj_ints)){
  loss_maj[i] <- loss_cvg(ytest[i], maj_ints[i,])
  loss_rand_maj[i] <- loss_cvg(as.numeric(ytest[i]), rand_maj_ints[i,])
}
mean(loss_maj); mean(loss_rand_maj)

# Example
i <- 100
vote <- cbind(loss_cvg(ys, sets1[i,]), loss_cvg(ys, sets2[i,]), loss_cvg(ys, sets3[i,]), loss_cvg(ys, sets4[i,]), 
              loss_cvg(ys, sets5[i,]), loss_cvg(ys, sets6[i,]), loss_cvg(ys, sets7[i,]))
colnames(vote) <- c("NN", "RF", "XG_Boost", "CART", "QDA", "MARS", "LDA")
rownames(vote) <- 1:nclass
vote_p <- as.data.frame(as.table(vote))
colnames(vote_p) <- c("Class", "Method", "Loss")
plot_v1 <- ggplot(vote_p, aes(x = Class, y = Method, fill = Loss)) +
           geom_tile() +
           scale_fill_gradient(low = "white", high = "blue") +  
           labs(title = "") +
           ylab("") +
           theme(legend.position = "bottom", legend.title = element_blank()) +
           theme_minimal()

maj_vote_i <- matrix(1, 2, nclass)
maj_vote_i[1,rowMeans(vote)<0.5] <- 0
set.seed(4)
u <- runif(1,0,0.5)
maj_vote_i[2,rowMeans(vote)<(0.5-u)] <- 0
maj_vote_i <- t(maj_vote_i)
colnames(maj_vote_i) <- c("Maj Vote", "Rand Maj Vote")
rownames(vote) <- 1:nclass
vote_pi <- as.data.frame(as.table(maj_vote_i))
colnames(vote_pi) <- c("Class", "Method", "Set")
vote_pi$Class <- rep(as.factor(1:10), 2)
plot_v2 <- ggplot(vote_pi, aes(x = Class, y = Method, fill = Set)) +
  geom_tile() +
  scale_fill_gradient(low = "darkred", high = "white") +  
  labs(title = "") +
  theme_minimal() +
  ylab("") +
  theme(legend.position = "none", legend.title = element_blank())
plot_v2

grid.arrange(plot_v1, plot_v2, ncol = 2)
