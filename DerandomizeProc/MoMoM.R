# MoM estimator
rm(list = ls())
library(sn)
library(ggplot2)

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

# An example ----
set.seed(1)
B <- 21      # number of buckets
L <- 1000    # number of sims
n <- 210     # number of obs
K <- 70      # max iter
res <- reps <- rep(NA, K)

x <- rt(n, df = 3)
for(k in 1:K){
  reps[k]<- MoM_fn(x, B)
  res[k] <- median(reps[1:k])
}

data.plot1 <- data.frame(K=1:K, mu=res, mu_mom=reps)
muplot1 <- ggplot(data.plot1, aes(x = K, y = mu)) +
           geom_point() +
           geom_hline(yintercept = 0, linetype = "dashed", color = "blue") + 
           labs(title = "MoMoM estimator (t with 3 d.f.)",
                x = "# of replications (K)",
                y = expression(hat(mu)^MoMoM)) +
           theme_minimal() +
           ylim( - 0.2, 0.2)
muplot1
mumom_plot1 <- ggplot(data.plot1, aes(x = K, y = mu_mom)) +
  geom_point() +
  geom_hline(yintercept = 0, linetype = "dashed", color = "blue") + 
  labs(title = "MoM estimator (t with 3 d.f.)",
       x = "# of replications (K)",
       y = expression(hat(mu)^MoM)) +
  theme_minimal() + 
  ylim(- 0.3, 0.3)
mumom_plot1

x <- rst(n, nu = 3, alpha = 1)
for(k in 1:K){
  reps[k]<- MoM_fn(x, B)
  res[k] <- median(reps[1:k])
}
a <- 1 #skew parm
delta <- a/sqrt(1+a^2)
mu <- delta * sqrt(3/pi) * gamma(1)/gamma(3/2)
data.plot2 <- data.frame(K=1:K, mu=res, mu_mom=reps)
muplot2 <- ggplot(data.plot2, aes(x = K, y = mu)) +
  geom_point() + 
  geom_hline(yintercept = mu, linetype = "dashed", color = "blue") + 
  labs(title = "MoMoM estimator (skew-t with 3 d.f.)",
       x = "# of replications (K)",
       y = expression(hat(mu)^MoMoM)) +
  theme_minimal() +
  ylim(0.6, 0.95)
muplot2
mumom_plot2 <- ggplot(data.plot2, aes(x = K, y = mu_mom)) +
  geom_point() + 
  geom_hline(yintercept = mu, linetype = "dashed", color = "blue") + 
  labs(title = "MoM estimator (skew-t with 3 d.f.)",
       x = "# of replications (K)",
       y = expression(hat(mu)^MoM)) +
  theme_minimal() +
  ylim(0.6, 0.95)
mumom_plot2

# Simulations 1 ------
## Simulate data from a t distribution -----
n <- 210     # number of obs
res <- reps <- matrix(NA, K, L)
for(i in 1:L){
  xs <- rt(n, 3)
  for(k in 1:K){
    reps[k,i]<- MoM_fn(xs, B)
    res[k,i] <- median(reps[1:k, i])
  }
  if(i %% 100 == 0) cat(i, "\n")
}

diff_one <- apply(res, 2, function(x) abs(diff(x)))
diff_one_t210 <- rowMeans(diff_one)

abs_t210 <- apply(res, 2, function(x) abs(x-0))
abs_t210 <- rowMeans(abs_t210)

## simulate data from a skew-t distribution -----
a <- 1       # skewness parm
n <- 210     # number of obs
res <- reps <- matrix(NA, K, L)
for(i in 1:L){
  xs <- rst(n, nu = 3, alpha = a)
  for(k in 1:K){
    reps[k,i]<- MoM_fn(xs, B)
    res[k,i] <- median(reps[1:k, i])
  }
  if(i %% 100 == 0) cat(i, "\n")
}

diff_one <- apply(res, 2, function(x) abs(diff(x)))
diff_one_st210 <- rowMeans(diff_one)

abs_st210 <- apply(res, 2, function(x) abs(x-mu))
abs_st210 <- rowMeans(abs_st210)

# Plot
data.plot1 <- cbind(1:(K-1), diff_one_t210, diff_one_st210)
colnames(data.plot1) <- c("K", "t", "st")
data.plot1 <- as.data.frame(data.plot1)
p1 <- ggplot(data.plot1, aes(x = K)) +
  geom_line(aes(y = t, color = "t (3 d.f.)"), linetype = "solid", linewidth = 1.2) +
  geom_line(aes(y = st, color = "Skew-t (3 d.f.)"), linetype = "dashed", linewidth = 1.2) +
  labs(title = "n=210",
       x = "# of replications (K)",
       y = expression(abs(hat(mu)[k]^MoMoM - hat(mu)[k-1]^MoMoM))) +
  theme_minimal() +
  scale_color_manual(values = c("t (3 d.f.)" = "blue", "Skew-t (3 d.f.)" = "red"),
                     name = "") +
  theme(legend.position = "bottom", legend.text = element_text(size = 12),
        title = element_text(size = 12), axis.title = element_text(size = 12)) + ylim(0, 0.045)
p1 


data.plot2 <- cbind(1:K, abs_t210, abs_st210)
colnames(data.plot2) <- c("K", "t", "st")
data.plot2 <- as.data.frame(data.plot2)
p2 <- ggplot(data.plot2, aes(x = K)) +
  geom_line(aes(y = t, color = "t (3 d.f.)"), linetype = "solid", linewidth = 1.2) +
  geom_line(aes(y = st, color = "Skew-t (3 d.f.)"), linetype = "dashed", linewidth = 1.2) +
  labs(title = "n=210",
       x = "# of replications (K)",
       y = expression(abs(hat(mu)[k]^MoMoM - mu))) +
  theme_minimal() +
  scale_color_manual(values = c("t (3 d.f.)" = "blue", "Skew-t (3 d.f.)" = "red"),
                     name = "") +
  theme(legend.position = "bottom", legend.text = element_text(size = 12),
        title = element_text(size = 12), axis.title = element_text(size = 12)) 
p2

# Final plots -----
plot0 <- gridExtra::grid.arrange(mumom_plot1, mumom_plot2, nrow = 2)
plot1 <- gridExtra::grid.arrange(muplot1, muplot2, nrow = 2)
plot2 <- gridExtra::grid.arrange(p1, p2, nrow = 2)
plotFN <- gridExtra::grid.arrange(plot0, plot1, plot2, ncol = 3)

