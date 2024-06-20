# MAJORITY VOTE

# Utils function
# counts (of voters) in a interval
# M: intervals
# a: lower
# b: upper
# w: weights
counts_int <- function(M, a, b, w){
  k <- ncol(M)
  num_int <- weighted.mean(I(M[,1]<=((a+b)*0.5) & ((a+b)*0.5)<=M[,2]), w)
  return(num_int)
}


# Function for majority vote
# M: matrix of interval (each interval lower and upper)
# w: weigths
# rho: threshold
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


# Other functions ------
len_ints <- function(M){
  # M: matrix
  if(sum(is.na(M))>0){
    return(0)
  }
  M <- matrix(M, ncol = 2)
  sum(M[,2]-M[,1])
}

cov_ints <- function(M, target){
  # M: matrix
  # target: integer
  if(sum(is.na(M))>0){
    return(0)
  }
  sum(apply(matrix(M, ncol = 2), 1, function(x) I(x[1] <= target & target <= x[2])))
}



