# Majority vote -----

# Given a matrix of containing the lower and the upper bounds of the intervals
# return a set using a majority vote procedure

merg.majority <- function(conf.ints, w, num.grid.pts = 1000){
  # conf.ints: matrix K x 2
  # w: weights
  # num.grid.points: number of points to test
  
  k <- nrow(conf.ints)
  w <- w/sum(w)
  
  ys.seq <- seq(min(conf.ints[,1]), max(conf.ints[,2]), length = num.grid.pts)
  
  vote   <- matrix(NA, nrow = num.grid.pts, ncol = k)
  for(i in 1:num.grid.pts){
    for(j in 1:k){
      vote[i,j] <- I(conf.ints[j,1] <= ys.seq[i] & ys.seq[i] <= conf.ints[j,2])
    }
  }
  
  vote.f   <- apply(vote, 1, function(x) I(sum(w*x) > 0.5))
  
  if(sum(vote.f) == 0){
    return(c(0,0))
  }
  else{
    which.lo <- min(which(vote.f==1))
    which.up <- max(which(vote.f==1))
  }
  return(c(ys.seq[which.lo], ys.seq[which.up]))  
}

# Randomized majority vote -----

# Given a matrix of containing the lower and the upper bounds of the intervals
# return a set using a randomized majority vote procedure

merg.majority.rand <- function(conf.ints, w, num.grid.pts = 1000){
  # conf.ints: matrix K x 2
  # w: weights
  # num.grid.points: number of points to test
  
  k <- nrow(conf.ints)
  w <- w/sum(w)
  a <- runif(1, 0, 0.5)
  
  ys.seq <- seq(min(conf.ints[,1]), max(conf.ints[,2]), length = num.grid.pts)
  
  vote   <- matrix(NA, nrow = num.grid.pts, ncol = k)
  for(i in 1:num.grid.pts){
    for(j in 1:k){
      vote[i,j] <- I(conf.ints[j,1] <= ys.seq[i] & ys.seq[i] <= conf.ints[j,2])
    }
  }
  
  vote.f   <- apply(vote, 1, function(x) I(sum(w*x) > 0.5 + a))
  
  if(sum(vote.f) == 0){
    return(c(0,0))
  }
  else{
    which.lo <- min(which(vote.f==1))
    which.up <- max(which(vote.f==1))
  }
  return(c(ys.seq[which.lo], ys.seq[which.up]))  
}

# Hedge Algorithm -----

# Given a matrix T x k of losses and a learning parameter returns the weigths 
# and the hedge loss over time

hedge <- function(l, eta){
  # l: matrix TxK containing the loss of the K experts during rounds
  # eta: learning parameter
  N <- nrow(l)
  K <- ncol(l)
  h <- rep(NA, N)
  L <- rep(0, K)
  
  weights <- matrix(NA, N, K)
  w <- rep(1/K, K)
  
  for(t in 1:N){
    weights[t,] <- w
    w <- w * exp(-eta * l[t,])
    w <- w / sum(w)
    h[t] <- sum(w * l[t,])
  }
  return(list(h=h, weights=weights))
}

# AdaHedge algorithm -----

# Given a matrix T x k of losses and returns the weigths and the hedge loss 
# over time of the adaHedge procedure

mix <- function(L, eta) {
  # eta: learning parm.
  # L: cumulative loss
  mn <- min(L)
  
  if (eta == Inf) {
    w <- as.numeric(L == mn)
  } else {
    w <- exp(-eta * (L - mn))
  }
  
  s <- sum(w)
  w <- w / s
  M <- mn - log(s / length(L)) / eta
  
  return(list(w = w, M = M))
}

adahedge <- function(l) {
  # l: matrix TxK containing the loss of the K experts during rounds
  N <- nrow(l)
  K <- ncol(l)
  h <- rep(NA, N)
  L <- rep(0, K)
  etas <- rep(NA, N)
  weights <- matrix(NA, N, K)
  Delta <- 0
  
  for (t in 1:N) {
    eta <- log(K) / Delta
    result <- mix(L, eta=eta)
    w <- result$w
    Mprev <- result$M
    h[t] <- sum(w * l[t,])
    weights[t,] <- w
    L <- L + l[t,]
    result <- mix(L, eta=eta)
    delta <- max(0, h[t] - (result$M - Mprev))
    Delta <- Delta + delta
    etas[t] <- eta
  }
  
  return(list("h"=h, "weights"=weights, "eta"=etas))
}

# Create matrix -----

# Create a (list of) matrix(ces) from a list obtained from conf.pred

create.matrix <- function(l){
  # l: list conf.pred
  k <- length(l)
  n <- length(l[[1]]$lo)
  list.mat <- vector("list", n)
  for(i in 1:n){
    mat.res <- matrix(NA, nrow = k, ncol = 2)
    for(j in 1:k){
      mat.res[j, 1] <- l[[j]]$lo[i]
      mat.res[j, 2] <- l[[j]]$up[i]
    }
    list.mat[[i]] <- mat.res
  }
  if(n == 1){
    return(list.mat[[1]])
  }
  else{
    return(list.mat)
  }
}



