library(MASS)
library(foreach)
library(randomForest)
library(doParallel)
library(isotree)

source("functions.R")



dataori <- rbind(read.csv('sat_train.csv', header = T), read.csv('sat_test.csv', header = T))
data_clean <- dataori[dataori$label!=2, ]
data_out <- dataori[dataori$label==2, ]



n <- 400
n1 <- 100
m <- 1000
pi0 <- 0.95
pi1 <- 0.05
alpha <- 0.1
d <- 36




nr <- 500
cl <- makeCluster(100)
registerDoParallel(cl)
Result <- foreach(iter = 1:nr, .combine = "rbind", .packages = c("MASS", "isotree", "randomForest"), .errorhandling = "remove")%dopar% {
  
  data <- data.frame()
  
  data_clean_use <- data_clean[sample(1:nrow(data_clean), nrow(data_clean)),]
  data_out_use <- data_out[sample(1:nrow(data_out), nrow(data_out)),]
  
  X0 <- data_clean_use[1:n, 1:d]
  X_test <- rbind(data_out_use[1:(pi1*m), 1:d], data_clean_use[(n+1):(n+pi0*m), 1:d])
  outlier <- 1:(pi1*m)
  X1 <- data_out_use[(pi1*m+1):(pi1*m+n1), 1:d]
  
  
  X0_train <- X0[1:(n/2),]
  X0_cal <- X0[(n/2+1):n,]
  X1_train <- X1[1:(n1/2),]
  X1_cal <- X1[(n1/2+1):n1,]
  
  
  model0_IOF <- isolation.forest(X0_train)
  model1_IOF <- isolation.forest(X1_train)
  model0_new <- isolation.forest(rbind(X0, X_test))
  
  model_RF <- randomForest(y~., data = data.frame(x = rbind(X0_train, X0_cal, X_test), y = factor(c(rep(0, n/2), rep(1, n/2+m)))), ntree = 500)
  model_RF_B <- randomForest(y~., data = data.frame(x = rbind(X0_train, X1), y = factor(c(rep(0, n/2), rep(1, n1)))), ntree = 500)
  model_RF_new <- randomForest(y~., data = data.frame(x = rbind(X0, X_test, X1), y = factor(c(rep(0, n+m), rep(1, n1)))), ntree = 500)
  
  #model_RF_B <- svm(y~., data = data.frame(x = rbind(X0_train, X1), y = factor(c(rep(0, n/2), rep(1, n1)))), probability = T)
  #model_RF_new <- svm(y~., data = data.frame(x = rbind(X0, X_test, X1), y = factor(c(rep(0, n+m), rep(1, n1)))), probability = T)
  
  
  s0_cal <- predict(model0_IOF, X0_cal)
  s0_cal1 <- predict(model1_IOF, X0_cal)
  s1_cal <- predict(model1_IOF, X1_cal)
  s0_new_cal1 <- predict(model1_IOF, X0)
  s0_test <- predict(model0_IOF, X_test)
  s1_test <- predict(model1_IOF, X_test)
  s0_new <- predict(model0_new, X_test)
  s0_new_cal <- predict(model0_new, X0)
  
  V_cal <- predict(model_RF, data.frame(x = X0_cal), type = 'prob')[, 2]
  V_test <- predict(model_RF, data.frame(x = X_test), type = 'prob')[, 2]
  VB_cal <- predict(model_RF_B, data.frame(x = X0_cal), type = 'prob')[, 2]
  VB_test <- predict(model_RF_B, data.frame(x = X_test), type = 'prob')[, 2]
  Vn_cal <- predict(model_RF_new, data.frame(x = X0), type = 'prob')[, 2]
  Vn_test <- predict(model_RF_new, data.frame(x = X_test), type = 'prob')[, 2]
  
  #VB_cal <- attr(predict(model_RF_B, data.frame(x = X0_cal), probability = T), 'probabilities')[, 2]
  #VB_test <- attr(predict(model_RF_B, data.frame(x = X_test), probability = T), 'probabilities')[, 2]
  #Vn_cal <- attr(predict(model_RF_new, data.frame(x = X0), probability = T), 'probabilities')[, 2]
  #Vn_test <- attr(predict(model_RF_new, data.frame(x = X_test), probability = T), 'probabilities')[, 2]
  
  
  
  pval_cp <- sapply(s0_test, function(x){(sum(x<=s0_cal)+1)/(length(s0_cal)+1)})
  rej_cp <- BH(pval_cp, alpha)
  power_cp <- sum(rej_cp%in%outlier)/length(outlier)
  FDP_cp <- sum(!rej_cp%in%outlier)/max(length(rej_cp), 1)
  data <- rbind(data, data.frame(FDP = FDP_cp, POWER = power_cp, method = 'CP-oc'))
  
  
  pval_oc <- sapply(s0_new, function(x){(sum(x<=s0_new_cal)+1)/(length(s0_new_cal)+1)})
  rej_oc <- BH(pval_oc, alpha)
  power_oc <- sum(rej_oc%in%outlier)/length(outlier)
  FDP_oc <- sum(!rej_oc%in%outlier)/max(length(rej_oc), 1)
  data <- rbind(data, data.frame(FDP = FDP_oc, POWER = power_oc, method = 'FullND'))
  
  
  pval_ada <- sapply(V_test, function(x){(sum(x<=V_cal)+1)/(length(V_cal)+1)})
  rej_ada <- BH(pval_ada, alpha)
  power_ada <- sum(rej_ada%in%outlier)/length(outlier)
  FDP_ada <- sum(!rej_ada%in%outlier)/max(length(rej_ada), 1)
  data <- rbind(data, data.frame(FDP = FDP_ada, POWER = power_ada, method = 'AdaDet'))
  
  
  pval_BIN <- sapply(VB_test, function(x){(sum(x<=VB_cal)+1)/(length(VB_cal)+1)})
  rej_BIN <- BH(pval_BIN, alpha)
  power_BIN <- sum(rej_BIN%in%outlier)/length(outlier)
  FDP_BIN <- sum(!rej_BIN%in%outlier)/max(length(rej_BIN), 1)
  data <- rbind(data, data.frame(FDP = FDP_BIN, POWER = power_BIN, method = 'CP-bi'))
  
  
  pval_new <- sapply(Vn_test, function(x){(sum(x<=Vn_cal)+1)/(length(Vn_cal)+1)})
  rej_new <- BH(pval_new, alpha)
  power_new <- sum(rej_new%in%outlier)/length(outlier)
  FDP_new <- sum(!rej_new%in%outlier)/max(length(rej_new), 1)
  data <- rbind(data, data.frame(FDP = FDP_new, POWER = power_new, method = 'ECOT-bi'))
  
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
  data <- rbind(data, data.frame(FDP = FDP_int, POWER = power_int, method = 'Integ'))
  
  
  
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
  data <- rbind(data, data.frame(FDP = FDP_int_re, POWER = power_int_re, method = 'ECOT-oc'))

  s0_sum <- sapply(s0_new_cal, function(x){sum(x<=s0_new_cal)})
  s1_sum <- sapply(s0_new_cal1, function(x){sum(x>=s1_cal)}) + 1
  pval_int_re2 <- rep(0, m)
  for (i in 1:m) {
    u0 <- c(s0_sum + ifelse(s0_new[i]>=s0_new_cal, 1, 0), sum(s0_new[i]<=s0_new_cal) + 1) / (n+1)
    u1 <- c(s1_sum, sum(s1_test[i]>=s1_cal) + 1) / (n1/2+1)
    r <- u0/u1
    pval_int_re2[i] <- sum(r[n+1]>=r)/(n+1)
  }
  
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
  
  
  ###---method selection---###
  pval_mat <- cbind(pval_new, pval_int_re1, pval_int_re2, pval_int_re3, pval_int_re4, pval_oc)
  optvec <- c('bi', 'oc++', 'oc+-', 'oc-+', 'oc--', 'full')
  rejnum_sel <- rep(NA, m)
  opt_method <- rep(NA, m)
  pval_sel <- rep(NA, m)
  for (j in 1:m){
    rem <- which.max(c(s0_new_cal, s0_new[j]))
    rem_bi <- which.max(c(Vn_cal, Vn_test[j]))
      
    rejnum_each <- rep(NA, 6)
    Vn_cal_sel <- c(Vn_cal, Vn_test[j])
    Vn_cal_sel <- Vn_cal_sel[-rem_bi]
    Vn_test_sel <- Vn_test[-j]
    pval_new_sel <- sapply(Vn_test_sel, function(x){(sum(x<=Vn_cal_sel)+1)/(length(Vn_cal_sel)+1)})
      if(j==1){
        pval_new_sel <- c(0, pval_new_sel)
      }else if(j==m){
        pval_new_sel <- c(pval_new_sel, 0)
      }else {
        pval_new_sel <- c(pval_new_sel[1:(j-1)], 0, pval_new_sel[j:(m-1)])
      }
      pval_new_sel_mod <- sapply(pval_new_sel, function(x){x-1/(length(Vn_cal_sel)+1)})
      pval_new_sel_mod[j] <- 0
      rejnum_each[1] <- length(BH(pval_new_sel, alpha))
      
      
    s0_new_cal_sel <- c(s0_new_cal, s0_new[j])
    s0_new_cal_sel <- s0_new_cal_sel[-rem]
    s0_new_sel <- s0_new[-j]
    s1_test_sel <- s1_test[-j]
      ###++
      s0_sum_sel <- sapply(s0_new_cal_sel, function(x){sum(x<=s0_new_cal_sel)})
      s1_sum_sel <- sapply(c(s0_new_cal1, s1_test[j])[-rem], function(x){sum(x<=s1_cal)}) + 1
      pval_int_re_sel1 <- rep(NA, m-1)
      for (i in 1:(m-1)) {
        u0_sel <- c(s0_sum_sel + ifelse(s0_new_sel[i]>=s0_new_cal_sel, 1, 0), sum(s0_new_sel[i]<=s0_new_cal_sel) + 1) / (length(s0_new_cal_sel)+1)
        u1_sel <- c(s1_sum_sel, sum(s1_test_sel[i]<=s1_cal) + 1) / (n1/2+1)
        r_sel <- u0_sel/u1_sel
        pval_int_re_sel1[i] <- sum(r_sel[length(s0_new_cal_sel)+1]>=r_sel)/(length(s0_new_cal_sel)+1)
      }
      if(j==1){
        pval_int_re_sel1 <- c(0, pval_int_re_sel1)
      }else if(j==m){
        pval_int_re_sel1 <- c(pval_int_re_sel1, 0)
      }else {
        pval_int_re_sel1 <- c(pval_int_re_sel1[1:(j-1)], 0, pval_int_re_sel1[j:(m-1)])
      }
      pval_int_re_sel_mod1 <- sapply(pval_int_re_sel1, function(x){x-1/(length(s0_new_cal_sel)+1)})
      pval_int_re_sel_mod1[j] <- 0
      rejnum_each[2] <- length(BH(pval_int_re_sel1, alpha))
      
      ###+-
      s0_sum_sel <- sapply(s0_new_cal_sel, function(x){sum(x<=s0_new_cal_sel)})
      s1_sum_sel <- sapply(c(s0_new_cal1, s1_test[j])[-rem], function(x){sum(x>=s1_cal)}) + 1
      pval_int_re_sel2 <- rep(NA, m-1)
      for (i in 1:(m-1)) {
        u0_sel <- c(s0_sum_sel + ifelse(s0_new_sel[i]>=s0_new_cal_sel, 1, 0), sum(s0_new_sel[i]<=s0_new_cal_sel) + 1) / (length(s0_new_cal_sel)+1)
        u1_sel <- c(s1_sum_sel, sum(s1_test_sel[i]>=s1_cal) + 1) / (n1/2+1)
        r_sel <- u0_sel/u1_sel
        pval_int_re_sel2[i] <- sum(r_sel[length(s0_new_cal_sel)+1]>=r_sel)/(length(s0_new_cal_sel)+1)
      }
      if(j==1){
        pval_int_re_sel2 <- c(0, pval_int_re_sel2)
      }else if(j==m){
        pval_int_re_sel2 <- c(pval_int_re_sel2, 0)
      }else {
        pval_int_re_sel2 <- c(pval_int_re_sel2[1:(j-1)], 0, pval_int_re_sel2[j:(m-1)])
      }
      pval_int_re_sel_mod2 <- sapply(pval_int_re_sel2, function(x){x-1/(length(s0_new_cal_sel)+1)})
      pval_int_re_sel_mod2[j] <- 0
      rejnum_each[3] <- length(BH(pval_int_re_sel2, alpha))
      
      ###-+
      s0_sum_sel <- sapply(s0_new_cal_sel, function(x){sum(x>=s0_new_cal_sel)})
      s1_sum_sel <- sapply(c(s0_new_cal1, s1_test[j])[-rem], function(x){sum(x<=s1_cal)}) + 1
      pval_int_re_sel3 <- rep(NA, m-1)
      for (i in 1:(m-1)) {
        u0_sel <- c(s0_sum_sel + ifelse(s0_new_sel[i]<=s0_new_cal_sel, 1, 0), sum(s0_new_sel[i]>=s0_new_cal_sel) + 1) / (length(s0_new_cal_sel)+1)
        u1_sel <- c(s1_sum_sel, sum(s1_test_sel[i]<=s1_cal) + 1) / (n1/2+1)
        r_sel <- u0_sel/u1_sel
        pval_int_re_sel3[i] <- sum(r_sel[length(s0_new_cal_sel)+1]>=r_sel)/(length(s0_new_cal_sel)+1)
      }
      if(j==1){
        pval_int_re_sel3 <- c(0, pval_int_re_sel3)
      }else if(j==m){
        pval_int_re_sel3 <- c(pval_int_re_sel3, 0)
      }else {
        pval_int_re_sel3 <- c(pval_int_re_sel3[1:(j-1)], 0, pval_int_re_sel3[j:(m-1)])
      }
      pval_int_re_sel_mod3 <- sapply(pval_int_re_sel3, function(x){x-1/(length(s0_new_cal_sel)+1)})
      pval_int_re_sel_mod3[j] <- 0
      rejnum_each[4] <- length(BH(pval_int_re_sel3, alpha))
      
      ###--
      s0_sum_sel <- sapply(s0_new_cal_sel, function(x){sum(x>=s0_new_cal_sel)})
      s1_sum_sel <- sapply(c(s0_new_cal1, s1_test[j])[-rem], function(x){sum(x>=s1_cal)}) + 1
      pval_int_re_sel4 <- rep(NA, m-1)
      for (i in 1:(m-1)) {
        u0_sel <- c(s0_sum_sel + ifelse(s0_new_sel[i]<=s0_new_cal_sel, 1, 0), sum(s0_new_sel[i]>=s0_new_cal_sel) + 1) / (length(s0_new_cal_sel)+1)
        u1_sel <- c(s1_sum_sel, sum(s1_test_sel[i]>=s1_cal) + 1) / (n1/2+1)
        r_sel <- u0_sel/u1_sel
        pval_int_re_sel4[i] <- sum(r_sel[length(s0_new_cal_sel)+1]>=r_sel)/(length(s0_new_cal_sel)+1)
      }
      if(j==1){
        pval_int_re_sel4 <- c(0, pval_int_re_sel4)
      }else if(j==m){
        pval_int_re_sel4 <- c(pval_int_re_sel4, 0)
      }else {
        pval_int_re_sel4 <- c(pval_int_re_sel4[1:(j-1)], 0, pval_int_re_sel4[j:(m-1)])
      }
      pval_int_re_sel_mod4 <- sapply(pval_int_re_sel4, function(x){x-1/(length(s0_new_cal_sel)+1)})
      pval_int_re_sel_mod4[j] <- 0
      rejnum_each[5] <- length(BH(pval_int_re_sel4, alpha))
      
      
      ##full
      pval_oc_sel <- sapply(s0_new_sel, function(x){(sum(x<=s0_new_cal_sel)+1)/(length(s0_new_cal_sel)+1)})
      pval_oc_sel_mod <- sapply(pval_oc_sel, function(x){x-1/(length(s0_new_cal_sel)+1)})
      pval_oc_sel_mod[j] <- 0
      rejnum_each[6] <- length(BH(pval_oc_sel, alpha))
      
      
      rejnum_each_mod <- rep(NA, 6)
      rejnum_each_mod[1] <- length(BH(pval_new_sel_mod, alpha))
      rejnum_each_mod[2] <- length(BH(pval_int_re_sel_mod1, alpha))
      rejnum_each_mod[3] <- length(BH(pval_int_re_sel_mod2, alpha))
      rejnum_each_mod[4] <- length(BH(pval_int_re_sel_mod3, alpha))
      rejnum_each_mod[5] <- length(BH(pval_int_re_sel_mod4, alpha))
      rejnum_each_mod[6] <- length(BH(pval_oc_sel_mod, alpha))
      if(sum(rejnum_each==1)<6){
        if(sum(rejnum_each==max(rejnum_each))==1){
          ind <- which(rejnum_each==max(rejnum_each))
          opt_method[j] <- optvec[ind]
        }else {
          if(sum(rejnum_each_mod==max(rejnum_each_mod))==1){
            ind <- which(rejnum_each_mod==max(rejnum_each_mod))
            opt_method[j] <- paste('mod~', optvec[ind])
          }else {
            ind <- sample(which(rejnum_each_mod==max(rejnum_each_mod)), 1)
            opt_method[j] <- paste('rand~mod~', optvec[ind])
          }
        }
      }else {
        if(sum(rejnum_each_mod==max(rejnum_each_mod))==1){
          ind <- which(rejnum_each_mod==max(rejnum_each_mod))
          opt_method[j] <- paste('mod~', optvec[ind])
        }else {
          ind <- sample(which(rejnum_each_mod==max(rejnum_each_mod)), 1)
          opt_method[j] <- paste('rand~mod~', optvec[ind])
        }
      }
      
      rejnum_sel[j] <- max(rejnum_each)
      pval_sel[j] <- pval_mat[j, ind]
  }
  rej_sel_init <- which(pval_sel<=alpha*rejnum_sel/m)
  if(length(rej_sel_init)>=max(rejnum_sel)){
    rej_sel <- rej_sel_init
  }else {
    rej_sel <- rej_sel_init[BH(runif(length(rej_sel_init))*rejnum_sel[rej_sel_init]/length(rej_sel_init), 1)]
  }
  rej_sel <- BH(pval_sel, alpha)
  power_sel <- sum(rej_sel%in%outlier)/length(outlier)
  FDP_sel <- sum(!rej_sel%in%outlier)/max(length(rej_sel), 1)
  data <- rbind(data, data.frame(FDP = FDP_sel, POWER = power_sel, method = 'ECOT-as'))
  
  return(data)
}
stopCluster(cl)

save(Result, file = 'Satellite.RData')

