library(MASS)
library(foreach)
library(randomForest)
library(doParallel)
library(e1071)
library(isotree)
library(dplyr)

source("functions.R")



m <- 1000
pi0 <- 0.95
alpha <- 0.1
d <- 50
pi0l <- 0.8


nfull <- 400
sig <- 1
pi0array <- c(0.8, 0.825, 0.85, 0.875, 0.9, 0.925, 0.95)


nr <- 500
cl <- makeCluster(10)
registerDoParallel(cl)
Result <- foreach(iter = 1:nr, .combine = "rbind", .packages = c("MASS", "isotree", "e1071", "randomForest"), .errorhandling = "remove")%dopar% {
  
  data <- data.frame()
  
  for (pi0 in pi0array) {
    n <- ceiling(nfull*pi0l/2)*2
    n1 <- ceiling((nfull - n)/2)*2
    
    X0 <- mvrnorm(n, rep(0, d), diag(d))
    X1 <- mvrnorm(n1, c(rep(sig*sqrt(log(d)), 5), rep(0, d-5)), diag(d))
    X_test <- rbind(mvrnorm(pi0*m, rep(0, d), diag(d)), mvrnorm(m-pi0*m, c(rep(sig*sqrt(log(d)), 5), rep(0, d-5)), diag(d)))
    outlier <- (pi0*m+1):m
    
    
    X0_train <- X0[1:(n/2),]
    X0_cal <- X0[(n/2+1):n,]
    X1_train <- X1[1:(n1/2),]
    X1_cal <- X1[(n1/2+1):n1,]
    
    m_clean <- ceiling(m/2)
    
    
    model1_IOF <- isolation.forest(X1_train)
    model0_new <- isolation.forest(rbind(X0, X_test))
    
    V_idf <- predict(model0_new, rbind(X0, X_test))
    if(sum(V_idf==sort(V_idf)[n+m_clean])==1){
      X0_clean <- rbind(X0, X_test)[which(V_idf<=sort(V_idf)[n+m_clean]), ]
    }else{
      X0_clean <- rbind(X0, X_test)[c(which(V_idf<sort(V_idf)[n+m_clean]), sample(which(V_idf==sort(V_idf)[n+m_clean]), n+m_clean-length(which(V_idf<sort(V_idf)[n+m_clean])))), ]
    }
    model0_new_clean <- isolation.forest(X0_clean)
    
    s1_cal <- predict(model1_IOF, X1_cal)
    s0_new_cal1 <- predict(model1_IOF, X0)
    s1_test <- predict(model1_IOF, X_test)
    s0_new <- predict(model0_new, X_test)
    s0_new_cal <- predict(model0_new, X0)
    s0_new_clean <- predict(model0_new_clean, X_test)
    s0_new_cal_clean <- predict(model0_new_clean, X0)
    
    
    model_RF_new <- randomForest(y~., data = data.frame(x = rbind(X0, X_test, X1), y = factor(c(rep(0, n+m), rep(1, n1)))), ntree = 500)
    
    V_idf <- predict(model_RF_new, data.frame(x = rbind(X0, X_test)), type = 'prob')[, 2]
    if(sum(V_idf==sort(V_idf)[n+m_clean])==1){
      X0_clean <- rbind(X0, X_test)[which(V_idf<=sort(V_idf)[n+m_clean]), ]
    }else{
      X0_clean <- rbind(X0, X_test)[c(which(V_idf<sort(V_idf)[n+m_clean]), sample(which(V_idf==sort(V_idf)[n+m_clean]), n+m_clean-length(which(V_idf<sort(V_idf)[n+m_clean])))), ]
    }
    model_RF_new_clean <- randomForest(y~., data = data.frame(x = rbind(X0_clean, X1), y = factor(c(rep(0, n+m_clean), rep(1, n1)))), ntree = 500)
    
    
    Vn_cal <- predict(model_RF_new, data.frame(x = X0), type = 'prob')[, 2]
    Vn_test <- predict(model_RF_new, data.frame(x = X_test), type = 'prob')[, 2]
    Vn_cal_clean <- predict(model_RF_new_clean, data.frame(x = X0), type = 'prob')[, 2]
    Vn_test_clean <- predict(model_RF_new_clean, data.frame(x = X_test), type = 'prob')[, 2]
    
    
    pval_new <- sapply(Vn_test, function(x){(sum(x<=Vn_cal)+1)/(length(Vn_cal)+1)})
    rej_new <- BH(pval_new, alpha)
    power_new <- sum(rej_new%in%outlier)/length(outlier)
    FDP_new <- sum(!rej_new%in%outlier)/max(length(rej_new), 1)
    data <- rbind(data, data.frame(FDP = FDP_new, POWER = power_new, method = 'ECOT-bi', n = nfull, sig = sig, alpha = alpha, pi0 = pi0))
    
    pval_new_clean <- sapply(Vn_test_clean, function(x){(sum(x<=Vn_cal_clean)+1)/(length(Vn_cal_clean)+1)})
    rej_new_clean <- BH(pval_new_clean, alpha)
    power_new_clean <- sum(rej_new_clean%in%outlier)/length(outlier)
    FDP_new_clean <- sum(!rej_new_clean%in%outlier)/max(length(rej_new_clean), 1)
    data <- rbind(data, data.frame(FDP = FDP_new_clean, POWER = power_new_clean, method = 'ECOT-bi-pure', n = nfull, sig = sig, alpha = alpha, pi0 = pi0))
    
    s0_sum <- sapply(s0_new_cal, function(x){sum(x<=s0_new_cal)})
    s1_sum <- sapply(s0_new_cal1, function(x){sum(x<=s1_cal)}) + 1
    pval_int_re1 <- rep(0, m)
    for (i in 1:m) {
      u0 <- c(s0_sum + ifelse(s0_new[i]>=s0_new_cal, 1, 0), sum(s0_new[i]<=s0_new_cal) + 1) / (n+1)
      u1 <- c(s1_sum, sum(s1_test[i]<=s1_cal) + 1) / (n1/2+1)
      r <- u0/u1
      pval_int_re1[i] <- sum(r[n+1]>=r)/(n+1)
    }
    rej_int_re <- BH(pval_int_re1, alpha)
    power_int_re <- sum(rej_int_re%in%outlier)/length(outlier)
    FDP_int_re <- sum(!rej_int_re%in%outlier)/max(length(rej_int_re), 1)
    data <- rbind(data, data.frame(FDP = FDP_int_re, POWER = power_int_re, method = 'ECOT-oc', n = nfull, sig = sig, alpha = alpha, pi0 = pi0))
    
    s0_sum_clean <- sapply(s0_new_cal_clean, function(x){sum(x<=s0_new_cal_clean)})
    s1_sum <- sapply(s0_new_cal1, function(x){sum(x<=s1_cal)}) + 1
    pval_int_re1_clean <- rep(0, m)
    for (i in 1:m) {
      u0 <- c(s0_sum_clean + ifelse(s0_new_clean[i]>=s0_new_cal_clean, 1, 0), sum(s0_new_clean[i]<=s0_new_cal_clean) + 1) / (n+1)
      u1 <- c(s1_sum, sum(s1_test[i]<=s1_cal) + 1) / (n1/2+1)
      r <- u0/u1
      pval_int_re1_clean[i] <- sum(r[n+1]>=r)/(n+1)
    }
    rej_int_re_clean <- BH(pval_int_re1_clean, alpha)
    power_int_re_clean <- sum(rej_int_re_clean%in%outlier)/length(outlier)
    FDP_int_re_clean <- sum(!rej_int_re_clean%in%outlier)/max(length(rej_int_re_clean), 1)
    data <- rbind(data, data.frame(FDP = FDP_int_re_clean, POWER = power_int_re_clean, method = 'ECOT-oc-pure', n = nfull, sig = sig, alpha = alpha, pi0 = pi0))
  }
  
  return(data)
}
stopCluster(cl)

save(Result, file = 'CleanComp.RData')


pp <- Result%>%
  group_by(method, alpha, n, sig, pi0)%>%
  dplyr::summarize(FDR = mean(FDP), power = mean(POWER), sdFDR = sd(FDP)/sqrt(nr), sdpower = sd(POWER)/sqrt(nr))
pp
