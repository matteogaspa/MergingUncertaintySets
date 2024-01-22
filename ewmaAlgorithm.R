# Hedge Algorithm
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
  weights <- matrix(NA, N, K)
  Delta <- 0
  
  for (t in 1:N) {
    eta <- log(K) / Delta
    result <- mix(L, eta)
    w <- result$w
    Mprev <- result$M
    h[t] <- sum(w * l[t,])
    weights[t,] <- w
    L <- L + l[t,]
    result <- mix(L, eta)
    delta <- max(0, h[t] - (result$M - Mprev))
    Delta <- Delta + delta
  }
  
  return(list("h"=h, "weights"=weights))
}


