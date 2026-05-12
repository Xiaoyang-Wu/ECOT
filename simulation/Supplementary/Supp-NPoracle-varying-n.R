library(MASS)
library(foreach)
library(randomForest)
library(doParallel)
library(isotree)


source("functions.R")


m <- 1000
pi0 <- 0.95
alpha <- 0.1
d <- 50


sig <- 1.5
narray <- seq(100, 30100, 5000)


nr <- 500
cl <- makeCluster(10)
registerDoParallel(cl)
Result <- foreach(iter = 1:nr, .combine = "rbind", .packages = c("MASS", "isotree", "randomForest"), .errorhandling = "remove")%dopar% {
  
  data <- data.frame()
  
  for (nfull in narray) {
    n <- nfull*0.8
    n1 <- nfull*0.2
    
    X0 <- mvrnorm(n, rep(0, d), diag(d))
    X1 <- mvrnorm(n1, c(rep(sig*sqrt(log(d)), 5), rep(0, d-5)), diag(d))
    X_test <- rbind(mvrnorm(pi0*m, rep(0, d), diag(d)), mvrnorm(m-pi0*m, c(rep(sig*sqrt(log(d)), 5), rep(0, d-5)), diag(d)))
    outlier <- (pi0*m+1):m
    
    
    X0_train <- X0[1:(n/2),]
    X0_cal <- X0[(n/2+1):n,]
    X1_train <- X1[1:(n1/2),]
    X1_cal <- X1[(n1/2+1):n1,]
    
    
    model0_IOF <- isolation.forest(X0_train)
    model0_new <- isolation.forest(rbind(X0, X_test))
    
    model_RF <- randomForest(y~., data = data.frame(x = rbind(X0_train, X0_cal, X_test), y = factor(c(rep(0, n/2), rep(1, n/2+m)))), ntree = 500)
    
    
    s0_cal <- predict(model0_IOF, X0_cal)
    s0_test <- predict(model0_IOF, X_test)
    s0_new <- predict(model0_new, X_test)
    s0_new_cal <- predict(model0_new, X0)
    
    V_cal <- predict(model_RF, data.frame(x = X0_cal), type = 'prob')[, 2]
    V_test <- predict(model_RF, data.frame(x = X_test), type = 'prob')[, 2]
    
    V_cal_ora <- X0%*%c(rep(sig*sqrt(log(d)), 5), rep(0, d-5))
    V_test_ora <- X_test%*%c(rep(sig*sqrt(log(d)), 5), rep(0, d-5))
    
    pval_ora <- sapply(V_test_ora, function(x){(sum(x<=V_cal_ora)+1)/(length(V_cal_ora)+1)})
    rej_ora <- BH(pval_ora, alpha)
    power_ora <- sum(rej_ora%in%outlier)/length(outlier)
    FDP_ora <- sum(!rej_ora%in%outlier)/max(length(rej_ora), 1)
    data <- rbind(data, data.frame(FDP = FDP_ora, POWER = power_ora, method = 'NP', n = nfull, sig = sig))

    pval_cp <- sapply(s0_test, function(x){(sum(x<=s0_cal)+1)/(length(s0_cal)+1)})
    rej_cp <- BH(pval_cp, alpha)
    power_cp <- sum(rej_cp%in%outlier)/length(outlier)
    FDP_cp <- sum(!rej_cp%in%outlier)/max(length(rej_cp), 1)
    data <- rbind(data, data.frame(FDP = FDP_cp, POWER = power_cp, method = 'CP-oc', n = nfull, sig = sig))
    
    
    pval_oc <- sapply(s0_new, function(x){(sum(x<=s0_new_cal)+1)/(length(s0_new_cal)+1)})
    rej_oc <- BH(pval_oc, alpha)
    power_oc <- sum(rej_oc%in%outlier)/length(outlier)
    FDP_oc <- sum(!rej_oc%in%outlier)/max(length(rej_oc), 1)
    data <- rbind(data, data.frame(FDP = FDP_oc, POWER = power_oc, method = 'FullND', n = nfull, sig = sig))
    
    
    pval_ada <- sapply(V_test, function(x){(sum(x<=V_cal)+1)/(length(V_cal)+1)})
    rej_ada <- BH(pval_ada, alpha)
    power_ada <- sum(rej_ada%in%outlier)/length(outlier)
    FDP_ada <- sum(!rej_ada%in%outlier)/max(length(rej_ada), 1)
    data <- rbind(data, data.frame(FDP = FDP_ada, POWER = power_ada, method = 'AdaDet', n = nfull, sig = sig))
    
    
  }
  
  return(data)
}
stopCluster(cl)

save(Result, file = 'NPn.RData')


