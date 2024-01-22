require(nnet)
require(rpart)

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


