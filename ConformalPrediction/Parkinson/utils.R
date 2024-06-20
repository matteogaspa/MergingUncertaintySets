require(nnet)
require(rpart)
require(conformalInference)

# nnet ------
nnet.train <- function(x, y, ...){
  nnet(x, y, ...)
}

nnet.preds <- function(out, newx, ...){
  predict(out, newx, ...)
}

nnet.funs <- list(
  train = nnet.train,
  predict = nnet.preds
)

# rpart -----
rpart.train <- function(x, y, ...){
  df.train <- cbind(y, x)
  rpart(y ~ ., data = df.train, ...)
}

rpart.preds <- function(out, newx, ...){
  predict(out, newx, ...)
}

# Majority vote ----
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


# Others functions ----
loss_fun <- function(c, sets, a, b){
  # c: target point
  # sets: list of set
  # a: parm
  size_int <- lapply(sets, function(x) x$up - x$lo) %>% unlist()
  cov_int  <- lapply(sets, function(x) c(x$lo <= c & c <= x$up)) %>% unlist()
  loss_vec <- a * size_int - b * cov_int
  return(loss_vec)
}


sizes <- function(sets){
  # sets: list of sets
  lapply(sets, function(x) x$up - x$lo) %>% unlist()
}


