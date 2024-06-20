# obtain lhat
get_lhat <- function(calib_loss_table, lambdas, alpha, B=1) {
  n <- nrow(calib_loss_table)
  rhat <- colMeans(calib_loss_table)
  lhat_idx <- max(which(((n/(n+1)) * rhat + B/(n+1) ) >= alpha) + 1, 0)
  # Return the corresponding lambda value at lhat_idx
  return(lambdas[lhat_idx])
}

# obtain calib_loss_table
get_loss_table  <- function(lambda, predictions, ycal){
  loss <-rep(NA, nrow(predictions))
  true_pred <- as.numeric(ycal)
  for(i in 1:nrow(predictions)){
    loss[i] <- ifelse(predictions[i,true_pred[i]] > 1-lambda, 0, (8+as.numeric(ycal))/18)
  }
  return(loss)
}

# loss 
loss_cvg <- function(y, set){
  return(ifelse(set[y] == 1, 0, (8+as.numeric(y))/18))
}
loss_cvg <- Vectorize(loss_cvg, "y")

# obtain the intervals with the method in Xu, Guo, Wei (2023)
get_intervals <- function(preds, ly, lambda, ycal){
  n <- nrow(preds)
  k <- ncol(preds)
  ss <- preds
  res <- matrix(NA, n, k)
  loss <- rep(NA, n)
  for(i in 1:n){
    sum_p <- 0
    u <- l <- which.max(ss[i,])
    while((sum_p < (1-lambda)) & (u-l < k)){
      if((l==1) & (u<k)){
        u <- u+1
        sum_p <- sum_p + ss[i,u]
      } else if((u==k) & (l>1)){
        l <- l-1
        sum_p <- sum_p + ss[i,l]
      } else if(ss[i,l-1]>=ss[i,u+1]){
        l <- l-1
        sum_p <- sum_p + ss[i,l]
      } else{
        u <- u+1
        sum_p <- sum_p + ss[i,u]
      }
    }
    res[i,] <- ifelse(seq_along(1:k) >= l & seq_along(1:k) <= u, 1, 0)
    loss[i] <- loss_cvg(ycal[i], res[i,])
    if(i %% 100 == 0) cat(i, "\n")
  }
  return(list(intervals = res, loss = c(loss)))
}
