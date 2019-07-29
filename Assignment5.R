rm(list=ls())
library(rpart)
library(RColorBrewer)
library(pROC)
library(randomForest)
library(permute)
library(adabag)
library(dplyr)
library(lava)
library(caret)
library(ROCR)

load("/home/hugh/stat_ML/Assignment6/data_project_backpain.RData")

set.seed(100)

class(dat)
# shuffle dataset
dat = dat[sample(nrow(dat)),]

dat$PainDiagnosis = as.factor(dat$PainDiagnosis)


get_train_split = function(df){
      # split data set into train, validate, test sets (80:10:10)
      # note: with a 70:15:15 split, there was not enough data for the log
      # regression to converge so this split was used instead.
      train_index = sample(1:nrow(df), size = nrow(df)*0.8)
      df_train = df[train_index,]
      test_index = setdiff(1:nrow(df), train_index)
      df_test = df[test_index,]
      validate_index = sample(1:nrow(df_test), size=nrow(df_test)*0.5)
      df_validate = df_test[validate_index,]
      test_index = setdiff(1:nrow(df_test), validate_index)
      df_test = df_test[test_index,]
      
      return(list("train"=train_index, "test"=test_index, "validate"=validate_index))
}

split = get_train_split(dat)

# fitting models
# note: due to the lack of data, the log regression doesn't always converge properly
# so results may vary.
fit_log <- glm(PainDiagnosis ~ ., data = dat, family = "binomial", subset=split$train)
fit_tree <- rpart(PainDiagnosis ~ ., data = dat, subset = split$train)
fit_rf <- randomForest(PainDiagnosis ~ ., data=dat, subset=split$train, importance=TRUE)
fit_bag <- bagging(PainDiagnosis ~ ., data=dat, subset=split$train)
fit_knn <- knn3(PainDiagnosis ~ ., data = dat, subset=split$train)

summary(fit_log)
coefs = summary(fit_log)$coefficients
coefs
rownames(coefs)[which(coefs[,4] < 0.1)]
# important variables: PainLocation, C4, C6, C8, C9
summary(fit_tree)
# important variables: C8, C10, C13, C26, C28, C33, PainLocation
pal = brewer.pal(10, "Paired")
varImpPlot(fit_rf, type=1, n.var = 10, col=pal)
# important variables: PainLocation, C8, C9, C13, C26, C2

ctrl = rfeControl(functions=rfFuncs, method="repeatedcv", repeats=5)
lmprofile = rfe(x=dat[,2:32], y=dat[,1], rfeControl=ctrl)
lmprofile
plot(lmprofile, type="l")
# According to the above 3 algorithms, the most important variables are:
# PainLocation, C8, C10, C13, C9 among other less important variables

# Decision trees are highly susceptible to overfitting so for this reason it
# is wise to limit the input dimensions to hopefully aid in future generalization

lean_df = select(dat, PainDiagnosis, PainLocation, Criterion8, Criterion10, Criterion13,
       Criterion26, Criterion9, Criterion4, Criterion6, Criterion33,
       Criterion2, Criterion28)


#####################################
#      Train, Validate, Test        #
#####################################

split = get_train_split(lean_df)

fit_log <- glm(PainDiagnosis ~ ., data = lean_df, family = "binomial", subset=split$train)
fit_tree <- rpart(PainDiagnosis ~ ., data = lean_df, subset = split$train)
fit_rf <- randomForest(PainDiagnosis ~ ., data=lean_df, subset=split$train, importance=TRUE)
fit_bag <- bagging(PainDiagnosis ~ ., data=lean_df, subset=split$train)
fit_knn <- knn3(PainDiagnosis ~ ., data = lean_df, subset=split$train)

model_fit = list("log"=fit_log, "tree"=fit_tree, "randomForest"=fit_rf, "bag"=fit_bag, "knn"=fit_knn)

model_predict = list()
for(name in names(model_fit)){
  predict = predict(model_fit[name], newdata = lean_df[split$validate,])
  model_predict = c(model_predict, c(name = predict))
}
# bagging confusion matrix
tab = model_predict$name.bag$confusion
tab
bagging_ac = sum(revdiag(tab))/sum(tab)
bagging_ac

tree_predict_class = c()
for(i in 1:nrow(model_predict$name.tree)){
  if(model_predict$name.tree[i, 1] > model_predict$name.tree[i, 2]){
    tree_predict_class = c(tree_predict_class, "Noiceptive")
  }
  else{
    tree_predict_class = c(tree_predict_class, "Neuropathic")
  }
}
tree_predict_class

tab <- table(lean_df$PainDiagnosis[split$validate], tree_predict_class)
tab
tree_ac = sum(revdiag(tab))/sum(tab)
tree_ac

model_predict$name.randomForest
tab <- table(lean_df$PainDiagnosis[split$validate], model_predict$name.randomForest)
tab
rf_ac = sum(diag(tab))/sum(tab)
rf_ac

model_predict$name.knn
knn_predict_class = c()
for(i in 1:nrow(model_predict$name.knn)){
  if(model_predict$name.knn[i, 1] > model_predict$name.knn[i, 2]){
    knn_predict_class = c(knn_predict_class, "Noiceptive")
  }
  else{
    knn_predict_class = c(knn_predict_class, "Neuropathic")
  }
}

tab <- table(lean_df$PainDiagnosis[split$validate], knn_predict_class)
tab
tree_ac = sum(revdiag(tab))/sum(tab)
tree_ac

# random forest regression had the highest accuracy along with
# the knn on the validation
# set with a perfect classifcation score therefor I will use and fully
# evaluate this model from now on

prediction = predict(fit_rf, newdata=lean_df[split$test,])
tab <- table(lean_df$PainDiagnosis[split$test], prediction)
tab
rf_ac = sum(diag(tab))/sum(tab)
rf_ac

###############################
#       K-Fold Analysis       #
###############################

# test was initially performed with 1000 iterations
# results may be different from the report
iterations = 10
av_accuracy = 0
iter_ac_rf = numeric(iterations)
iter_recall_rf = numeric(iterations)
iter_precision_rf = numeric(iterations)
iter_ac_knn = numeric(iterations)
iter_recall_knn = numeric(iterations)
iter_precision_knn = numeric(iterations)
k = 5
n = nrow(lean_df)
folds = rep(1:k, ceiling(n/k))
folds = sample(folds)
folds = folds[1:n]

get_metrics <- function(tab){
  tab <- table(lean_df$PainDiagnosis[split$test], prediction)
  precision = tab[1]/(tab[1]+tab[3])
  recall = tab[1]/(tab[1]+tab[2])
  ac = sum(diag(tab))/sum(tab)
  output = list("ac"=ac, "recall"=recall, "precision"=precision)
  
}

for(j in 1:iterations){
  total_accuracy_rf = 0
  total_recall_rf = 0
  total_precision_rf = 0
  total_accuracy_knn = 0
  total_recall_knn = 0
  total_precision_knn = 0
  
  for(i in 1:k){
    train = which(folds != i)
    test = setdiff(1:n, train)
    
    fit_knn = knn3(PainDiagnosis ~ ., data = lean_df, subset = train)
    fit_rf = randomForest(PainDiagnosis ~ ., data = lean_df, subset = train)
    
    # get metrics for random forest
    prediction = predict(fit_rf, newdata=lean_df[split$test,])
    tab <- table(lean_df$PainDiagnosis[split$test], prediction)
    rf_metrics = get_metrics(tab)
    
    #get metrics for knn
    prediction = predict(fit_knn, type="class", newdata=lean_df[split$test,])
    tab <- table(lean_df$PainDiagnosis[split$test], prediction)
    knn_metrics = get_metrics(tab)
    
    # append metrics to respective running totals
    total_accuracy_rf = total_accuracy_rf + rf_metrics$ac
    total_recall_rf = total_recall_rf + rf_metrics$recall
    total_precision_rf = total_precision_rf + rf_metrics$precision
    
    total_accuracy_knn = total_accuracy_knn + knn_metrics$ac
    total_recall_knn = total_recall_knn + knn_metrics$recall
    total_precision_knn = total_precision_knn + knn_metrics$precision
  }
  
  iter_ac_rf[j] = total_accuracy_rf/k
  iter_recall_rf[j] = total_recall_rf/k
  iter_precision_rf[j] = total_precision_rf/k
  
  iter_ac_knn[j] = total_accuracy_knn/k
  iter_recall_knn[j] = total_recall_knn/k
  iter_precision_knn[j] = total_precision_knn/k
}

mean(iter_ac_rf)
mean(iter_recall_rf)
mean(iter_precision_rf)
2*(mean(iter_precision_rf)*mean(iter_recall_rf))/(mean(iter_precision_rf)+mean(iter_recall_rf))

mean(iter_ac_knn)
mean(iter_recall_knn)
mean(iter_precision_knn)
2*(mean(iter_precision_knn)*mean(iter_recall_knn))/(mean(iter_precision_knn)+mean(iter_recall_knn))

print_table = function(ac1, ac2){
  
  #calculate averages across all iterations
  av_ac1 = mean(ac1)
  av_ac2 = mean(ac2)
  
  max_acc = max(max(ac1), max(ac2))
  min_acc = min(min(ac1), min(ac2))
  
  sd_ac1 = sd(ac1)
  sd_ac2 = sd(ac2)
  
  #get upper and lower bounds for both models (2 standard deviations)
  ac1_ubound = av_ac1 + 2*sd_ac1
  ac1_lbound = av_ac1 - 2*sd_ac1
  
  ac2_ubound = av_ac2 + 2*sd_ac2
  ac2_lbound = av_ac2 - 2*sd_ac2
  
  max_sd = max(max(ac1_ubound), max(ac2_ubound))
  min_sd = min(min(ac1_lbound), min(ac2_lbound))
  
  pal = brewer.pal(3, "Spectral")
  
  plot(ac1, type = "l", col=pal[3], ylim = c(min_sd,max_sd),
       xlab = "Iterations", ylab = "Accuracy (%)")  
  
  lines(ac2, type = "l", col = pal[1])
  
  lines(c(0, iterations), c(av_ac1, av_ac1), type="l", col=pal[3])
  lines(c(0, iterations), c(av_ac2, av_ac2), type="l", col=pal[1])
  lines(c(0, iterations), c(ac1_ubound, ac1_ubound), type="l", col=pal[3], lty=2)
  lines(c(0, iterations), c(ac1_lbound, ac1_lbound), type="l", col=pal[3], lty=2)
  lines(c(0, iterations), c(ac2_ubound, ac2_ubound), type="l", col=pal[1], lty=2)
  lines(c(0, iterations), c(ac2_lbound, ac2_lbound), type="l", col=pal[1], lty=2)
  
  par(xpd=TRUE)
  legend(x=0, min_acc-0.0001, legend = c("KNN", "Random Forest"), col = c(pal[1], pal[3]), lty=1:1, cex=0.8)
  
  return(0)
}

print_table(iter_ac_rf, iter_ac_knn)

# overall, very high precision and slightly lower recall
# higher recall likely more desirable for cases such as this where
# a False negative can lead to serious health issues down the line
# in a sense it is better to predict positive more often than usual
# so as to avoid these 'expensive' misclassifications opposed
# to misclassifying a False Positive, in which case the only cost is
# the cost of treatment and not someone's long term health

###################################
#       Plotting ROC Curve        #
###################################
split = get_train_split(lean_df)
fit_rf <- randomForest(PainDiagnosis ~ ., data=lean_df, importance=TRUE)
pred = predict(fit_rf, newdata=lean_df)
pal = brewer.pal(3, "Spectral")
plot.roc(roc_curve, col=pal, print.auc=TRUE, identity.col=pal[3])

predObj <- prediction(as.numeric(pred), dat$PainDiagnosis, label.ordering=c("Nociceptive","Neuropathic"))

perf <- performance(predObj, "tpr", "fpr")
auc <- performance(predObj, "auc")@y.values
plot(perf, col=pal, identity.col=pal[3])
abline(0,1,lty=2,col=pal[3])
text(locator(), labels=auc


