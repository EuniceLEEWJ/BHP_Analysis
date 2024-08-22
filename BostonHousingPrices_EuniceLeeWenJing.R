### QUESTION 1 ###
#----------------------------------------------------------------------------
# 1.1
setwd("C:/Users/User/OneDrive - Monash University/Desktop/Projects/BHP_Analysis")
#1.1
rm(list=ls())

housing_data = read.csv("housing.2023.csv", header=TRUE)
fit = lm(medv ~., data=housing_data)
summary(fit)


#1.2
p_values = coefficients(summary(fit))[,4]
p_values
0.05/12 #0.004166667
p_values < 0.05/12
sum(p_values < 0.05/12)
# The Bonferroni threshold of 0.05/12 = 0.004166667
# 5 predictors have a p-value small enough to pass (6-1 excluding intercept) 
# Using the Bonferroni procedure, the predictors possibly associated are 
# chas, rm, dis, ptratio, lstat.

#1.3
# coefficient for crim 
fit$coefficients[[2]]
# coefficient for chas 
fit$coefficients[[5]]

#1.4
fit_bic = step(fit, k = log(length(housing_data$medv)),direction="both")
summary(fit_bic)
fit_bic$coefficients

#1.5

#1.6

new_housing <- data.frame(
  crim = 0.04741,
  zn = 0,
  indus = 11.93,
  chas = 0,
  nox = 0.573,
  rm = 6.03,
  age = 80.8,
  dis = 2.505,
  rad = 1,
  tax = 273,
  ptratio = 21,
  lstat = 7.88
)
summary(new_housing)

# Predict function to find the median house price for new housing
# the lwr and upr = the confidence interval 
new_house <- predict(fit_bic,new_housing , interval = "confidence")
new_house

#1.7
model <- lm(medv ~ rm*dis, data = housing_data)
summary(model)

### QUESTION 2 ###
#----------------------------------------------------------------------------
setwd("C:/Users/User/OneDrive - Monash University/Desktop/FIT2086/Assignment/A3")
source("my.prediction.stats.R")
source("wrappers.R")
install.packages("glmnet")
library(glmnet)
install.packages("rpart")
library(rpart)
install.packages("randomForest")
library(randomForest)
install.packages("kknn")
library(kknn)
library(boot)
library(rpart.plot)

# 2.1
heart.train = read.csv("heart.train.2023.csv",stringsAsFactors=T)
# fit decision tree to data
tree.hd = rpart(HD ~ ., heart.train)
tree.hd
plot(tree.hd, uniform = TRUE, branch = 1)
text(tree.hd, digits=3) #output 3 significant val
# cross validation
cv.tree.hd = learn.tree.cv(HD~.,data=heart.train, nfolds=10, m=5000)
cv.tree.hd
# The CV plot shows the best tree size is 7.
plot.tree.cv(cv.tree.hd)
# Find the variables use in best tree
cv.tree.hd$best.tree
plot(cv.tree.hd$best.tree)
text(cv.tree.hd$best.tree, digits=3)

# 2.2
plot(cv.tree.hd$best.tree)
text(cv.tree.hd$best.tree, pretty=12)

# 2.3
cv.tree.hd$best.tree

# 2.4

# 2.5
# fit logistic regression model to data
fullmod.hd = glm(HD ~ ., data=heart.train, family=binomial)
summary(fullmod.hd)
# stepwise selection with BIC score 
bic.hd = step(fullmod.hd, k = log(length(heart.train$HD)))
summary(bic.hd)

# 2.6

# 2.7
install.packages("pROC")
library(pROC)
heart.test = read.csv("heart.test.2023.csv",stringsAsFactors=T)

# prediction stat for cv pruned tree
my.pred.stats(predict(cv.tree.hd$best.tree,heart.test)[,2],heart.test$HD)
# prediction stat for step-wise logistic regression model
my.pred.stats(predict(bic.hd,heart.test,type="response"),heart.test$HD)

# 2.8
# conditional probabilities of 69th patient using cv pruned tree
predict(cv.tree.hd$best.tree,heart.test[69,])
# odds using tree
odd = 0.8636364/0.1363636 
odd #6.333335

# conditional probabilities of 10th patient using logistic regression model
predict(bic.hd,heart.test[69,],type="response")
# odds using logistic regression model
odd_1= 0.9463509/(1 - 0.94635093)
odd_1 

# 2.9
# 69th patient logistic regression
boot.auc = function(formula, data, indices)
{
  # Create a bootstrapped version of our data
  d = data[indices,]
  
  # Fit a logistic regression to the bootstrapped data
  fit = glm(formula, d, family=binomial)
  
  target= heart.test[69,]
  # Compute the AUC and return it
  rv = predict(fit,target,type="response")
  return(rv)
}
bs = boot(data=heart.train, statistic=boot.auc, R=5000, formula=HD ~ CP+THALACH +OLDPEAK +CA +THAL)
boot.ci(bs,conf=0.95,type="bca")
plot(bs)

### QUESTION 3 ###
#----------------------------------------------------------------------------
setwd("C:/Users/User/OneDrive - Monash University/Desktop/FIT2086/Assignment/A3")

source("wrappers.R")
library(glmnet)
library(rpart)
library(randomForest)
library(kknn)
library(boot)

ms_measured <- read.csv("ms.measured.2023.csv")
ms_truth <- read.csv("ms.truth.2023.csv")

# 3.1
k <- 1:25
mse <- c()
for(i in k) {
  ytest.hat = fitted(kknn(intensity ~ MZ, ms_measured, ms_truth, kernel = "optimal", k =i) )
  mse_of_each_k = mean((ytest.hat - ms_truth$intensity)^2)
  mse = c(mse,mse_of_each_k)
}
mse

# plot 
plot(mse~k, lwd=2, xlab="k-values", ylab="Errors", main="Errors for each k-values {1,...,25}")

# 3.2 
# k=2
# Training data points (ms.measured$intensity)
plot(x=ms_measured$MZ, y=ms_measured$intensity, xlab="Mass/Charge (MZ)", ylab="Relative Intensity", 
     main="Comparison between training data points, true spectrum and estimated spectrum (k=2)", pch=16, cex = 0.5,  col="blue")
# True spectrum  (ms.truth$intensity) 
lines(x=ms_truth$MZ, y=ms_truth$intensity, lwd=1,  type="l", col="orange")
# Predicted intensity values
lines(x=ms_truth$MZ, y=fitted(kknn(intensity~., ms_measured, ms_truth, kernel="optimal", k=2) ), lwd=1, type="l",col="red")
# Legend
legend("topright", legend = c("Training data points", "True spectrum", "Estimated spectrum"),
       lwd = 2, col = c("blue", "orange", "red"))
# k=5
# Training data points (ms.measured$intensity)
plot(x=ms_measured$MZ, y=ms_measured$intensity, xlab="Mass/Charge (MZ)", ylab="Relative Intensity", 
     main="Comparison between training data points, true spectrum and estimated spectrum (k=5)", pch=16, cex = 0.5,  col="blue")
# True spectrum  (ms.truth$intensity) 
lines(x=ms_truth$MZ, y=ms_truth$intensity, lwd=1,  type="l", col="orange")
# Predicted intensity values
lines(x=ms_truth$MZ, y=fitted(kknn(intensity~., ms_measured, ms_truth, kernel="optimal", k=5) ), lwd=1, type="l",col="red")
# Legend
legend("topright", legend = c("Training data points", "True spectrum", "Estimated spectrum"),
       lwd = 2, col = c("blue", "orange", "red"))

# k=10
# Training data points (ms.measured$intensity)
plot(x=ms_measured$MZ, y=ms_measured$intensity, xlab="Mass/Charge (MZ)", ylab="Relative Intensity", 
     main="Comparison between training data points, true spectrum and estimated spectrum (k=10)", pch=16, cex = 0.5,  col="blue")
# True spectrum  (ms.truth$intensity) 
lines(x=ms_truth$MZ, y=ms_truth$intensity, lwd=1,  type="l", col="orange")
# Predicted intensity values
lines(x=ms_truth$MZ, y=fitted(kknn(intensity~., ms_measured, ms_truth, kernel="optimal", k=10) ), lwd=1, type="l",col="red")
# Legend
legend("topright", legend = c("Training data points", "True spectrum", "Estimated spectrum"),
       lwd = 2, col = c("blue", "orange", "red"))

# k=25
# Training data points (ms.measured$intensity)
plot(x=ms_measured$MZ, y=ms_measured$intensity, xlab="Mass/Charge (MZ)", ylab="Relative Intensity", 
     main="Comparison between training data points, true spectrum and estimated spectrum (k=25)", pch=16, cex = 0.5,  col="blue")
# True spectrum  (ms.truth$intensity) 
lines(x=ms_truth$MZ, y=ms_truth$intensity, lwd=1,  type="l", col="orange")
# Predicted intensity values
lines(x=ms_truth$MZ, y=fitted(kknn(intensity~., ms_measured, ms_truth, kernel="optimal", k=25) ), lwd=1, type="l",col="red")
# Legend
legend("topright", legend = c("Training data points", "True spectrum", "Estimated spectrum"),
       lwd = 2, col = c("blue", "orange", "red"))


# 3.3

mse_k2=mean((fitted(kknn(intensity~., ms_measured, ms_truth, kernel="optimal", k=2) ) - ms_truth$intensity)^2)
mse_k5=mean((fitted(kknn(intensity~., ms_measured, ms_truth, kernel="optimal", k=5) ) - ms_truth$intensity)^2)
mse_k10=mean((fitted(kknn(intensity~., ms_measured, ms_truth, kernel="optimal", k=10) ) - ms_truth$intensity)^2)
mse_k25=mean((fitted(kknn(intensity~., ms_measured, ms_truth, kernel="optimal", k=25) ) - ms_truth$intensity)^2)

mse_k2
mse_k5
mse_k10
mse_k25

# 3.5
knn = train.kknn(intensity ~ ., data = ms_measured, kmax=25, kernel="optimal")
knn$best.parameters$k

# 3.6
# k=6 is selected
ytest.hat1 = fitted(kknn(intensity ~ .,ms_measured, ms_truth,kernel = "optimal", k = 6) )
diff = ytest.hat1 - ms_measured$intensity
sd(diff)

# 3.7
# index in max peaks (k=6)
max_index_estimate = which.max(ytest.hat1)

# MZ for the index 
max_MZ_estimate = ms_truth$MZ[max_index_estimate]
max_MZ_estimate

max_index_estimate
max_MZ_estimate


# 3.8
boot.intensity = function(data, indices, k_val)
{
  # Create a bootstrapped version of our data
  d = data[indices,]
  
  # use k-NN to estimate the values of spectrum associated with MZ values 
  ytest.hat2 = fitted(kknn(intensity ~ ., d, ms_truth, kernel = "optimal", k = k_val) )  
  
  return(ytest.hat2[283])
}

# k=3
bs_intensity = boot(data=ms_measured, statistic=boot.intensity, R=5000, k_val=3)
boot.ci(bs_intensity,conf=0.95,type="bca")

# k=6
bs_intensity = boot(data=ms_measured, statistic=boot.intensity, R=5000, k_val=6)
boot.ci(bs_intensity,conf=0.95,type="bca")

# k=20
bs_intensity = boot(data=ms_measured, statistic=boot.intensity, R=5000, k_val=20)
boot.ci(bs_intensity,conf=0.95,type="bca")




