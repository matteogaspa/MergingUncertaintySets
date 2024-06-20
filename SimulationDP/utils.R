# majority vote
# counts in a interval 
counts_int <- function(M, a, b, w){
  # M: matrix
  k <- ncol(M)
  num_int <- weighted.mean(I(M[,1]<=((a+b)*0.5) & ((a+b)*0.5)<=M[,2]), w)
  return(num_int)
}

counts_set <- function(M, a, b){
  # M: list
  k <- length(M)
  vote <- rep(0, k)
  for(i in 1:k){
    if(is.na(M[[i]][1])){
      vote[i] <- 0
    }else{
      vote[i] <- sum(apply(M[[i]], 1, function(x) I(x[1]<=((a+b)*0.5) & ((a+b)*0.5)<=x[2])))
    }
  }
  num_int <- sum(vote)
  return(num_int)
}


# M: matrix of interval (each interval lower and upper)
# q: quantile
majority_vote <- function(M, w, rho=0.5){
  k <- nrow(M)
  breaks <- as.vector(M)
  breaks <- unique(breaks)
  breaks <- breaks[order(breaks)]
  i <- 1
  lower <- upper <- NULL
    
  while(i < length(breaks)) {
    cond <- (counts_int(M, breaks[i], breaks[i+1], w)>rho)
    if(cond){
      lower <- c(lower, breaks[i])
      j <- i
      while(j < length(breaks) & cond){
        j <- j+1
        cond <- counts_int(M, breaks[j], breaks[j+1], w)>rho
      }
      i <- j
      upper <- c(upper, breaks[i])
    }
    i <- i+1
  }
  if(is.null(lower)){
    return(NA)
  }else{
    return(cbind(lower, upper))
  }
}
 

# exchangeable majority vote
# M: matrix of interval (each interval lower and upper)
exch_majority_vote <- function(M, tau = 0.5){
  k <- nrow(M)
  if(k==1){ return(M) }
  perm <- sample(1:k, replace = F)
  permM <- M[perm,]
  newM <- vector("list", k)
  newM[[1]] <- matrix(permM[1,], ncol = 2)
  for(i in 2:k){
    newM[[i]] <- majority_vote(permM[1:i,], rep(1/i, i), tau)
    if(is.na(newM[[i]][1])) return(NA)
  }
  breaks <- unlist(c(newM))
  breaks <- unique(breaks)
  breaks <- breaks[order(breaks)]
  i <- 1
  lower <- upper <- NULL
  
  while(i < length(breaks)) {
    cond <- ifelse(counts_set(newM, breaks[i], breaks[i+1])==k, T, F)
    if(cond){
      lower <- c(lower, breaks[i])
      j <- i
      while(j < length(breaks) & cond){
        j <- j+1
        cond <- (counts_set(newM, breaks[j], breaks[j+1])==k)
      }
      i <- j
      upper <- c(upper, breaks[i])
    }
    i <- i+1
  }
  if(is.null(lower)){
    return(NA)
  }else{
    return(cbind(lower, upper))
  }
}


# exchangeable randomized majority vote
exch_rand_majority_vote <- function(M){
  k <- nrow(M)
  if(k==1){ return(M) }
  perm <- sample(1:k, replace = F)
  permM <- M[perm,]
  newM <- vector("list", k)
  newM[[1]] <- matrix(permM[1,], ncol = 2)
  for(i in 2:(k-1)){
    newM[[i]] <- majority_vote(permM[1:i,], rep(1/i, i))
    if(is.na(newM[[i]][1])){ return(NA) }
  }
  newM[[k]] <- majority_vote(permM[1:k,], rep(1/k, k), runif(1, .5, 1))
  if(is.na(newM[[k]][1])){ return(NA) }
  
  breaks <- unlist(c(newM))
  breaks <- unique(breaks)
  breaks <- breaks[order(breaks)]
  i <- 1
  lower <- upper <- NULL
  
  while(i < length(breaks)) {
    cond <- ifelse(counts_set(newM, breaks[i], breaks[i+1])==k, T, F)
    if(cond){
      lower <- c(lower, breaks[i])
      j <- i
      while(j < length(breaks) & cond){
        j <- j+1
        cond <- (counts_set(newM, breaks[j], breaks[j+1])==k)
      }
      i <- j
      upper <- c(upper, breaks[i])
    }
    i <- i+1
  }
  if(is.null(lower)){
    return(NA)
  }else{
    return(cbind(lower, upper))
  }
}

