########################################Load Packages########################################

library(tidyverse)
library(caret)
library(mclust)
library(plotly)
library(dummies)

########################################Load Dataset########################################

india <- read.csv("indian_liver_patient.csv", na.strings = c("", "NA"), stringsAsFactors = FALSE)
india = india %>% dummy.data.frame(name = "Gender") %>% mutate(Dataset = ifelse(Dataset == 1, "Liver patient",
                                                                                ifelse(Dataset == 2, "Non-liver patient",NA)))
india$Dataset = factor(india$Dataset, levels = c("Liver patient", "Non-liver patient"))
india = india[complete.cases(india),]

########################################Demographic Characteristics########################################

india %>% mutate(Gender = ifelse(GenderMale == 1, "Male",
                                 ifelse(GenderFemale == 1, "Female",0))) %>%
  group_by(Dataset, Gender) %>% 
  summarise(SampleSize = n()) %>% group_by(Gender) %>% 
  mutate(Percent = SampleSize/sum(SampleSize) *100)

########################################Classification Ensemble ML########################################

#Create test and train sets#
set.seed(1)
ind <- createDataPartition(india$Dataset, times = 1, list = FALSE)
train <- india[-ind,]
trainset = train %>% select(-Dataset)

test <- india[ind,]
testset = test %>% select(-Dataset)

#Select models#
model <- c("svmLinear", "gbm", "svmRadial", "svmRadialCost", 
           "svmRadialSigma", "lda")
########################################Train########################################
#Apply all models to train set#
set.seed(1)
fit <- lapply(model, function(model){
  print(model)
  train(Dataset ~ ., method = model, data = train)
})
names(fit) <- model
########################################Predict########################################
#Predict test set from train fits#
pred <- sapply(fit, function(fit){
  predict(fit,test)
})
########################################Confusion Vector########################################
#Create confusion vector#
c <- c(1:ncol(pred))
confusionvector <- sapply(c, function(c){
  confusionMatrix(factor(pred[,c]), test$Dataset)$overall["Accuracy"]
})
mean(confusionvector)
cv = data.frame(Model=model, Accuracy=confusionvector)
########################################Majority Vote########################################
#Select majority vote#
c2 <- c(1:nrow(test))
mv <- sapply(c2, function(c2){
  m <- majorityVote(pred[c2,])
  m$majority
})
pred <- cbind(pred,mv)
########################################Confusion Vector Post Majority Vote########################################
#Create confusion matrix using majority vote#
cm <- confusionMatrix(factor(pred[,ncol(pred)]), test$Dataset)$overall["Accuracy"]
cm

######################################## PCA ########################################
pca = trainset %>% prcomp()
pca_sum = summary(pca)

########################################Plot of Primary Components########################################

#Plot of standard deviation by primary component#
plot(pca_sum$importance[3,], xlab="Primary Component", ylab="Cumulative Proportion")

#PC1, PC2 & PC3#
pca_train[,c(1,2,3,8)] %>% 
  gather(Component, Value, PC1:PC3, factor_key=TRUE) %>%
  ggplot(aes(Dataset, Value, fill = Component))+
  geom_point(cex=3, pch=21) +
  coord_cartesian(ylim=c(-1000, 1000)) + 
  labs(title = "Plot of Top 3 Primary Components and Diagnosis")

########################################PCA Test and Train Sets########################################
pca_train = data.frame(pca$x[,1:7], Dataset=train$Dataset)

pca_test = data.frame(prcomp(testset)$x[,1:7], Dataset=test$Dataset)
########################################Training Using PCA########################################
###Train using PCA train set###
pca_model <- c("svmLinear", "gbm", "svmRadial", "svmRadialCost", 
           "svmRadialSigma")

set.seed(1)
pca_fit <- lapply(pca_model, function(pca_model){
  print(pca_model)
  train(Dataset ~ ., method = pca_model, data = pca_train)
})
names(pca_fit) <- pca_model
########################################Predicting using PCA_Test########################################
#Predict test set from train fits#
pca_pred <- sapply(pca_fit, function(pca_fit){
  predict(pca_fit,pca_test)
})
########################################Confusion Vector Post-PCA########################################
#Create confusion vector#
pca_c <- c(1:ncol(pca_pred))
pca_confusionvector <- sapply(pca_c, function(pca_c){
  confusionMatrix(factor(pca_pred[,pca_c]), pca_test$Dataset)$overall["Accuracy"]
})
mean(pca_confusionvector)
pca_cv = data.frame(Model=pca_model, Accuracy=pca_confusionvector)
########################################PCA Majority Vote########################################
#Select majority vote#
pca_c2 <- c(1:nrow(pca_test))
pca_mv <- sapply(pca_c2, function(pca_c2){
  m <- majorityVote(pca_pred[pca_c2,])
  m$majority
})
pca_pred <- cbind(pca_pred,pca_mv)
########################################Confusion Vector after PCA Majority Vote########################################
#Create confusion matrix using majority vote#
pca_cm <- confusionMatrix(factor(pca_pred[,ncol(pca_pred)]), pca_test$Dataset)$overall["Accuracy"]
pca_cm

########################################BONUS 3-D PLOT!########################################
######PC1, PC2, & PC3#########

pca_train[,c(1,2,3,8)] %>%
  plot_ly(x=.$PC1, y=.$PC2, z=.$PC3, color=.$Dataset, colors = c("indianred", "turquoise"))





