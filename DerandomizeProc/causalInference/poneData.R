rm(list = ls())
library(dplyr)
library(DoubleML)
library(mlr3)
library(mlr3learners)
library(causaldata)
library(ggplot2)
source("majorityVote.R")

# read dataset ----- 
pone1 <- readxl::read_xlsx("pone.0164255.s004.xlsx")
pone2 <- readxl::read_xlsx("pone.0164255.s005.xlsx")
head(pone1); head(pone2)

# clean dataset ----
pone1 <- as.data.frame(pone1)
colnames(pone1) <- c("treat", "age", "geneder", "smoker", "hypertension", "dislipidemia", "angiotensin", "statin", "metformin", "dn", "fmd", "sbp", "dbp", "csbp", "augmentation index", "cardiac_index", "tpri", "lf/hf", "height", "weight", "bmi", "fpg", "hba1c", "iri", "pi", "pii", "cpr", "c-peptide", "glucagon", "ldl", "hdl", "tryglyceride", "rlp", "adiponectin", "sod activity", "pai", "nt-pro", "tnf-a", "hscrp", "ualb", "droms", "bap")
pone1 <- as_tibble(pone1)
pone1 <- pone1 %>% select(treat, hba1c, age, geneder, bmi, sbp, dbp, hypertension, ldl, hdl, dislipidemia, adiponectin, cpr, cardiac_index, pii, bap, droms, tryglyceride)
pone1 <- as.data.frame(pone1)
str(pone1)

pone2 <- as.data.frame(pone2)
colnames(pone2) <- c("treat", "age", "geneder", "fmd", "sbp", "dbp", "csbp", "augmentation index", "cardiac_index", "tpri", "lf/hf", "height", "weight", "bmi", "fpg", "hba1c", "iri", "pi", "pii", "cpr", "c-peptide", "glucagon", "ldl", "hdl", "tryglyceride", "rlp", "adiponectin", "sod activity", "pai", "nt-pro", "tnf-a", "hscrp", "ualb", "droms", "bap")
pone1$outcome <- as.numeric(pone2$hba1c) - pone1$hba1c 

pone1 <- na.omit(pone1)
pone1 <- pone1[,-which(colnames(pone1)=="hba1c")]
pone1$outcome <- as.numeric(pone1$outcome < -0.5)
prop.table(table(pone1$outcome, pone1$treat), 1)

# choosing regression and classification algorithms -----
ml_g = lrn("regr.ranger", num.trees = 40, mtry = 4, min.node.size = 2)
ml_m = lrn("classif.ranger", num.trees = 40, mtry = 4, min.node.size = 2)

obj_dml_data <- DoubleMLData$new(pone1, y_col="outcome", d_cols="treat")
dml_irm_obj <- DoubleMLIRM$new(obj_dml_data, ml_g, ml_m,n_folds = 4)
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
  obj_dml_data <- DoubleMLData$new(pone1, y_col="outcome", d_cols="treat")
  dml_irm_obj <- DoubleMLIRM$new(obj_dml_data, ml_g, ml_m,n_folds = 4)
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
  "upE" = cisE[,2],
  "lo"  = cis[,1],
  "up"  = cis[,2]
)

plot1 <- ggplot(data = data.plot, aes(x = k)) +
  geom_point(aes(y = lo, color = "Majority Vote")) +
  geom_point(aes(y = up, color = "Majority Vote")) +
  geom_segment(aes(x = k, xend = k, y = lo, yend = up), color = "black") +
  labs(title = "Confidence intervals for ATE estimator",
       x = "# of replications (K)",
       y = "Confidence Intervals") +
  scale_color_manual(values = c("Majority Vote" = "black", "Exch. Majority Vote" = "red")) +
  theme_minimal() + theme(legend.position = "none", legend.title = element_blank()) + ylim(-0.4, 0.6) 
plot1

plot2 <- ggplot(data = data.plot, aes(x = k)) +
  geom_point(aes(y = loM, color = "Majority Vote")) +
  geom_point(aes(y = upM, color = "Majority Vote")) +
  geom_segment(aes(x = k, xend = k, y = loM, yend = upM), color = "black") +
  labs(title = "Merged confidence intervals",
       x = "# of replications (K)",
       y = "Confidence Intervals") +
  scale_color_manual(values = c("Majority Vote" = "black", "Exch. Majority Vote" = "red")) +
  theme_minimal() + theme(legend.position = "none", legend.title = element_blank()) + ylim(-0.4, 0.6)
plot2

gridExtra::grid.arrange(plot1, plot2, ncol = 2)


