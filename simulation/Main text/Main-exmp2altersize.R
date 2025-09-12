library(MASS)
library(foreach)
library(randomForest)
library(doParallel)
library(isotree)


source("functions.R")
source("algoclass.R")





m <- 1000
pi0 <- 0.8
alpha <- 0.1
d <- 1000

n0 <- 800
n1array <- c(20, 100, 400, 800, 1600)

W <- matrix(runif(d*d, -3, 3), nrow = d)
a0 <- 1
a1 <- 1.5


nr <- 100
cl <- makeCluster(100)
registerDoParallel(cl)
Result <- foreach(iter = 1:nr, .combine = "rbind", .packages = c("MASS", "isotree", "randomForest"), .errorhandling = "remove")%dopar% {
  
  data <- data.frame()
  
  for (n1 in n1array) {
    n <- n0
    
    sig <- a1
    
    V0 <- mvrnorm(n, rep(0, d), diag(d))
    V1 <- mvrnorm(n1, rep(0, d), diag(d))
    V_test <- mvrnorm(m, rep(0, d), diag(d))
    X0 <- sqrt(a0)*V0 + W[sample(1:nrow(W), n, replace = T),]
    X1 <- sqrt(a1)*V1 + W[sample(1:nrow(W), n1, replace = T),]
    X_test <- rbind(sqrt(a0)*V_test[1:(pi0*m),] + W[sample(1:nrow(W), pi0*m, replace = T),], sqrt(a1)*V_test[(pi0*m+1):m,] + W[sample(1:nrow(W), m-pi0*m, replace = T),])
    outlier <- (pi0*m+1):m
    
    
    X0_train <- X0[1:(n/2),]
    X0_cal <- X0[(n/2+1):n,]
    X1_train <- X1[1:(n1/2),]
    X1_cal <- X1[(n1/2+1):n1,]
    
    
    model0_IOF <- isolation.forest(X0_train)
    model1_IOF <- isolation.forest(X1_train)
    model0_new <- isolation.forest(rbind(X0, X_test))
    
    s0_cal <- predict(model0_IOF, X0_cal)
    s0_cal1 <- predict(model1_IOF, X0_cal)
    s1_cal <- predict(model1_IOF, X1_cal)
    s0_new_cal1 <- predict(model1_IOF, X0)
    s0_test <- predict(model0_IOF, X_test)
    s1_test <- predict(model1_IOF, X_test)
    s0_new <- predict(model0_new, X_test)
    s0_new_cal <- predict(model0_new, X0)


    pval_cp <- sapply(s0_test, function(x){(sum(x<=s0_cal)+1)/(length(s0_cal)+1)})
    rej_cp <- BH(pval_cp, alpha)
    power_cp <- sum(rej_cp%in%outlier)/length(outlier)
    FDP_cp <- sum(!rej_cp%in%outlier)/max(length(rej_cp), 1)
    data <- rbind(data, data.frame(FDP = FDP_cp, POWER = power_cp, method = 'CP-oc', n1 = n1, sig = sig))
    
    
    pval_oc <- sapply(s0_new, function(x){(sum(x<=s0_new_cal)+1)/(length(s0_new_cal)+1)})
    rej_oc <- BH(pval_oc, alpha)
    power_oc <- sum(rej_oc%in%outlier)/length(outlier)
    FDP_oc <- sum(!rej_oc%in%outlier)/max(length(rej_oc), 1)
    data <- rbind(data, data.frame(FDP = FDP_oc, POWER = power_oc, method = 'FullND', n1 = n1, sig = sig))
    
    
    s0_sum <- sapply(s0_cal, function(x){sum(x<=s0_cal)})
    s1_sum <- sapply(s0_cal1, function(x){sum(x>=s1_cal)}) + 1
    pval_int <- rep(0, m)
    for (i in 1:m) {
      u0 <- c(s0_sum + ifelse(s0_test[i]>=s0_cal, 1, 0), sum(s0_test[i]<=s0_cal) + 1)
      u1 <- c(s1_sum, sum(s1_test[i]>=s1_cal) + 1)
      r <- u0/u1
      pval_int[i] <- sum(r[n/2+1]>=r)/(n/2+1)
    }
    rej_int <- BH(pval_int, alpha)
    power_int <- sum(rej_int%in%outlier)/length(outlier)
    FDP_int <- sum(!rej_int%in%outlier)/max(length(rej_int), 1)
    data <- rbind(data, data.frame(FDP = FDP_int, POWER = power_int, method = 'Integ', n1 = n1, sig = sig))
    
    
    
    s0_sum <- sapply(s0_new_cal, function(x){sum(x<=s0_new_cal)})
   s1_sum <- sapply(s0_new_cal1, function(x){sum(x<=s1_cal)}) + 1
   pval_int_re1 <- rep(0, m)
   for (i in 1:m) {
     u0 <- c(s0_sum + ifelse(s0_new[i]>=s0_new_cal, 1, 0), sum(s0_new[i]<=s0_new_cal) + 1) / (n+1)
     u1 <- c(s1_sum, sum(s1_test[i]<=s1_cal) + 1) / (n1/2+1)
     r <- u0/u1
     pval_int_re1[i] <- sum(r[n+1]>=r)/(n+1)
   }
   

   s0_sum <- sapply(s0_new_cal, function(x){sum(x<=s0_new_cal)})
   s1_sum <- sapply(s0_new_cal1, function(x){sum(x>=s1_cal)}) + 1
   pval_int_re2 <- rep(0, m)
   for (i in 1:m) {
     u0 <- c(s0_sum + ifelse(s0_new[i]>=s0_new_cal, 1, 0), sum(s0_new[i]<=s0_new_cal) + 1) / (n+1)
     u1 <- c(s1_sum, sum(s1_test[i]>=s1_cal) + 1) / (n1/2+1)
     r <- u0/u1
     pval_int_re2[i] <- sum(r[n+1]>=r)/(n+1)
   }
   rej_int_re <- BH(pval_int_re2, alpha)
   power_int_re <- sum(rej_int_re%in%outlier)/length(outlier)
   FDP_int_re <- sum(!rej_int_re%in%outlier)/max(length(rej_int_re), 1)
   data <- rbind(data, data.frame(FDP = FDP_int_re, POWER = power_int_re, method = 'ECOT-oc', n1 = n1, sig = sig))
  
  s0_sum <- sapply(s0_new_cal, function(x){sum(x>=s0_new_cal)})
  s1_sum <- sapply(s0_new_cal1, function(x){sum(x<=s1_cal)}) + 1
  pval_int_re3 <- rep(0, m)
  for (i in 1:m) {
    u0 <- c(s0_sum + ifelse(s0_new[i]<=s0_new_cal, 1, 0), sum(s0_new[i]>=s0_new_cal) + 1) / (n+1)
    u1 <- c(s1_sum, sum(s1_test[i]<=s1_cal) + 1) / (n1/2+1)
    r <- u0/u1
    pval_int_re3[i] <- sum(r[n+1]>=r)/(n+1)
  }
  
  s0_sum <- sapply(s0_new_cal, function(x){sum(x>=s0_new_cal)})
  s1_sum <- sapply(s0_new_cal1, function(x){sum(x>=s1_cal)}) + 1
  pval_int_re4 <- rep(0, m)
  for (i in 1:m) {
    u0 <- c(s0_sum + ifelse(s0_new[i]<=s0_new_cal, 1, 0), sum(s0_new[i]>=s0_new_cal) + 1) / (n+1)
    u1 <- c(s1_sum, sum(s1_test[i]>=s1_cal) + 1) / (n1/2+1)
    r <- u0/u1
    pval_int_re4[i] <- sum(r[n+1]>=r)/(n+1)
  }
    
    
    
  }
  
  return(data)
}
stopCluster(cl)

save(Result, file = 'Result_2_n1.RData')


