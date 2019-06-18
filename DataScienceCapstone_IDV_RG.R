###Load Packages and Dataset###

library(tidyverse)
library(caret)
library(mclust)
library(plotly)

india <- read.csv("indian_liver_patient.csv", na.strings = c("", "NA"), stringsAsFactors = FALSE)
india$Gender = factor(india$Gender, levels = c("Male", "Female"))
india$Dataset[india$Dataset == 1] = "Liver patient"
india$Dataset[india$Dataset == 2] = "Non-liver patient"
india$Dataset = factor(india$Dataset, levels = c("Liver patient", "Non-liver patient"))
india = india[complete.cases(india),]

###Demographic Characteristics###

india %>% group_by(Dataset, Gender) %>% 
  summarise(SampleSize = n()) %>% group_by(Gender) %>% 
  mutate(Proportion = SampleSize/sum(SampleSize) *100)

###Ensemble Classification ML###
#Create test and train sets#
set.seed(1)
ind <- createDataPartition(india$Dataset, times = 1, list = FALSE)
train <- india[-ind,]
test <- india[ind,]

#Select models#
model <- c("lda", "naive_bayes", "svmLinear", "qda", 
           "knn", "kknn", "rpart", "rf", "ranger", "wsrf", 
           "Rborist", "avNNet", "mlp", "monmlp",
           "adaboost", "gbm", "svmRadial", 
           "svmRadialCost", "svmRadialSigma")

#Apply all models to train set#
set.seed(1)
fit <- lapply(model, function(model){
  train(Dataset ~ ., method = model, data = train)
})
names(fit) <- model

#Predict test set from train fits#
pred <- sapply(fit, function(fit){
  predict(fit,test)
})

#Create confusion vector#
c <- c(1:ncol(pred))
confusionvector <- sapply(c, function(c){
  confusionMatrix(factor(pred[,c]), test$Dataset)$overall["Accuracy"]
})
mean(confusionvector)

#Select majority vote#
c2 <- c(1:nrow(test))
mv <- sapply(c2, function(c2){
  m <- majorityVote(pred[c2,])
  m$majority
})
pred <- cbind(pred,mv)

#Create confusion matrix using majority vote#
cv <- confusionMatrix(factor(pred[,ncol(pred)]), test$Dataset)$overall["Accuracy"]
cv

###Check correlation matrix for PCA###
t = india %>% .[,1:10] %>% dummy.data.frame(name = "Gender") %>% as.matrix()
cor(t)

###PCA###
pca = train %>% dummy.data.frame(name = "Gender") %>% .[,1:10] %>% prcomp()
summary(pca)

###Plot of Primary Components###
#PC1 & PC2 #
data.frame(pca$x[,1:2], Dataset=train$Dataset) %>% 
  ggplot(aes(PC1,PC2, fill = Dataset))+
  geom_point(cex=3, pch=21) +
  coord_fixed(ratio = 1)+
  labs(title = "Plot of PC1 and PC2")

#PC2 & PC3#
data.frame(pca$x[,2:3], Dataset=train$Dataset) %>% 
  ggplot(aes(PC2,PC3, fill = Dataset))+
  geom_point(cex=3, pch=21) +
  coord_fixed(ratio = 1)+
  labs(title = "Plot of PC2 and PC3")

#PC1, PC2, & PC3#
data.frame(pca$x[,1:3], Dataset=train$Dataset) %>%
  plot_ly(x=.$PC1, y=.$PC2, z=.$PC3, type="scatter3d", color=.$Dataset, colors = c("indianred", "turquoise"))

###Run iterations to remove columns###
b = c(1:10)
set.seed(1)
iter <- sapply(b, function(b){
  train_b = train[,-b]
  test_b = test[,-b]
  set.seed(1)
  fit <- lapply(model, function(model){
    train(Dataset ~ ., method = model, data = train_b)})
  names(fit) <- model
  pred <- sapply(fit, function(fit){
    predict(fit,test_b)})
  c <- c(1:ncol(pred))
  confusionvector <- sapply(c, function(c){
    confusionMatrix(factor(pred[,c]), test_b$Dataset)$overall["Accuracy"]})
  mean(confusionvector)
  c2 <- c(1:nrow(test_b))
  mv <- sapply(c2, function(c2){
    m <- majorityVote(pred[c2,])
    m$majority})
  pred <- cbind(pred,mv)
  confusionMatrix(factor(pred[,ncol(pred)]), test_b$Dataset)$overall["Accuracy"]
})

#Rename iterations#
names(iter) <- c("Age removed", "Gender removed", "Total_Bilirubin removed", "Direct_Bilirubin removed", 
                 "Alkaline_Phosphotase removed", "Alamine_Aminotransferase removed", "Aspartate_Aminotransferase removed",
                 "Total_Proteins removed", "Albumin removed", "Albumin_Globulin_ratio removed")
iter

###Predicting from cleaned dataset###
india_cleaned = india[,c(1,4,6,8,9,11)]
train_cleaned = india_cleaned[-ind,]
test_cleaned = india_cleaned[ind,]

set.seed(1)
fit_cleaned <- lapply(model, function(model){
  train(Dataset ~ ., method = model, data = train_cleaned)
})
names(fit_cleaned) <- model

pred_cleaned <- sapply(fit_cleaned, function(fit_cleaned){
  predict(fit_cleaned,test_cleaned)
})

c3 <- c(1:ncol(pred_cleaned))
confusionvector_cleaned <- sapply(c3, function(c3){
  confusionMatrix(factor(pred_cleaned[,c3]), test_cleaned$Dataset)$overall["Accuracy"]
})
mean(confusionvector_cleaned)

c4 <- c(1:nrow(test_cleaned))
mv_cleaned <- sapply(c4, function(c4){
  m <- majorityVote(pred[c4,])
  m$majority
})
pred_cleaned <- cbind(pred_cleaned,mv_cleaned)

cv_cleaned <- confusionMatrix(factor(pred_cleaned[,ncol(pred_cleaned)]), test_cleaned$Dataset)$overall["Accuracy"]
cv_cleaned








