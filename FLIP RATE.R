
# Load the dplyr library for subsetting data
# rpart for classification trees
# randomForest for random forest models
# caret for creating partitions training and testing
# ROCR for ROC curves

library(dplyr)
library(dummies)
library(rgl)
library(cluster)
library(fpc)
library(rpart)
library(caret)
library(ROCR)
library(randomForest)
library(e1071)
library(kernlab)
library(nnet)


set.seed(123)

# Read the model data
model_data <- read.table("model_data.csv", header=TRUE, row.names=1, sep=",",
                         colClasses=c("character", "numeric",
                                      rep("factor",4),"numeric", "factor",rep("numeric",7)))


str(model_data)


# View the data frame model_data

summary(model_data)


model_data <- model_data %>%
  dplyr::select(Age,
                Gender,
                PrimaryInsuranceCategory,
                Flipped,
                DRG01,
                BloodPressureUpper,
                BloodPressureLower,
                BloodPressureDiff,
                Pulse,
                PulseOximetry,
                Respirations,
                Temperature) %>%
  
  filter(Flipped %in% c(0,1))


# NA Values in model data
sum(is.na(model_data))
sum(is.na(c(model_data$Pulse,model_data$Temperature, model_data$Respirations)))

# Replacing the NA Values with Median
model_data$Respirations[is.na(model_data$Respirations)] <- median(model_data$Respirations, na.rm=TRUE)
model_data$Pulse[is.na(model_data$Pulse)] <- median(model_data$Pulse, na.rm=TRUE)
model_data$Temperature[is.na(model_data$Temperature)] <- median(model_data$Temperature, na.rm=TRUE)

sum(is.na(model_data$Pulse))
sum(is.na(model_data$Temperature))
sum(is.na(model_data$Respirations))

sum(is.na(model_data))

# Clustering the data
model_df <- dummy.data.frame(model_data,
                             names=c("Flipped",
                                     "Gender",
                                     "PrimaryInsuranceCategory","DRG01"))
head(model_df)

# Scale the data
model_df <- scale(model_df, center=TRUE, scale=TRUE)
head(model_df)

# k means
# Using the k means clustering
# Principal Component Analysis to identify characteristics of each cluster
model_kmeans <- kmeans(model_df, centers=3)
model_pca <- prcomp(model_df, retx=TRUE)

model_pca
plot(model_pca$x[,1:2], col=model_kmeans$cluster, pch=model_kmeans$cluster, main="K-means Cluster")

summary(model_pca$x[,1:2])
model_pca$rotation[,1:2]
plot3d(model_pca$x[,1:3], col=model_kmeans$cluster)

boxplot(model_pca$x[,1:3], main="Box-plot")

# Partition data
set.seed(123)
train_rows <- createDataPartition(model_data$Flipped, p = 0.7, list=FALSE)
train_model <- model_data[train_rows,]
test_model <- model_data[-train_rows,]

summary(train_model)
summary(train_model$Flipped)
summary(test_model$Flipped)

# Assigning weights for classes
My_model_weights <- numeric(nrow(train_model))
My_model_weights[train_model$Flipped == "0"] <- 1
My_model_weights[train_model$Flipped == "1"] <- 2

# Logistic Regression
# * Model 1 *
My_model_lr <- glm(Flipped ~ .,
                   data=train_model,
                   weights=My_model_weights,
                   family=binomial("logit"))


summary(My_model_lr)
my_model_lr_predict <- predict(My_model_lr,
                               newdata=test_model,
                               type="response")
# Confusion Matrix 1
my_model_lr_predict_class <- character(length(my_model_lr_predict))
my_model_lr_predict_class[my_model_lr_predict < 0.5] <- 0
my_model_lr_predict_class[my_model_lr_predict >= 0.5] <- 1

my_model_lr1<-table(test_model$Flipped, my_model_lr_predict_class)
my_model_lr1
# Misclassification rate
logistic<- 1-sum(diag(my_model_lr1))/sum(my_model_lr1)
logistic
# logistic =  0.3795181

# ROC 1
model_lr_predict <- predict(My_model_lr, test_model, type="response")
model_lr_pred <- prediction(model_lr_predict,
                            test_model$Flipped,
                            label.ordering=c("0", "1"))
model_lr_pref <- performance(model_lr_pred, "tpr", "fpr")
plot(model_lr_pref)


# *   Model 2 *
# Classification tree model
My_model_rpart <- rpart(Flipped ~ ., data=train_model, weights=My_model_weights)
My_model_rpart_predict <- predict(My_model_rpart, newdata=test_model, type="class")
# Confusion Matrix 2
My_model_rpart1<-table(test_model$Flipped, My_model_rpart_predict)
My_model_rpart1
# Misclassifiaction Rate
classification<- 1-sum(diag(My_model_rpart1))/sum(My_model_rpart1)
classification
# classifiaction = 0.3674699
# Important variable
My_model_rpart$variable.importance
# Importantvariable visualisation
impvar1 <- table(model_data$DRG01)
barplot(impvar1)
impvar2 <- table(model_data$BloodPressureLower)
barplot(impvar2)

# ROC 2
model_rpart_predict <- predict(My_model_rpart, test_model, type="prob")
model_rpart_pred <- prediction(model_rpart_predict[,2],
                               test_model$Flipped,
                               label.ordering=c(0, 1))
model_rpart_perf <- performance(model_rpart_pred, "tpr", "fpr")

plot(model_rpart_perf)

# * Model 3* #
# Random forest
my_model_rf <- randomForest(Flipped ~ .,
                            data = train_model,
                            classwt=c(2,1),
                            importance=TRUE)



my_model_rf$importance
my_model_predict_rf <- predict(my_model_rf, newdata=test_model, type="class")

# Confusion Matrix 3
(my_model_rf_confusion <- table(test_model$Flipped, my_model_predict_rf))

# Misclassification rate 3
RF <- 1-sum(diag(my_model_rf_confusion))/sum(my_model_rf_confusion)
RF
# RF = 0.4457831

# ROC 3
model_rf_predict <- predict(my_model_rf, test_model, type="prob")
model_rf_pred <- prediction(model_rf_predict[,2],
                            test_model$Flipped,
                            label.ordering=c(0, 1))
model_rf_perf <- performance(model_rf_pred, "tpr", "fpr")

plot(model_rf_perf)


# * Model 4 * #
# SVM
train_model_dummy <- dummy.data.frame(train_model, names=c("Gender","PrimaryInsuranceCategory","DRG01"))
train_model_preprocess <- preProcess(train_model_dummy)
train_model_numeric <- predict(train_model_preprocess, train_model_dummy)
test_model_dummy <- dummy.data.frame(test_model, names=c("Gender","PrimaryInsuranceCategory","DRG01"))
test_model_numeric <- predict(train_model_preprocess, test_model_dummy)
#-------------------------------------------------
levels(train_model_numeric$Flipped) <- c("not_flipped", "flipped")
model_model_svm <- train(Flipped ~ .,
                         data=train_model_numeric,
                         method="svmLinearWeights",
                         metric="ROC",
                         trControl=trainControl(classProbs=TRUE,
                                                summaryFunction=twoClassSummary))
model_model_svm

modelLookup("svmLinearWeights")

model_model_predict_svm <- predict(model_model_svm, newdata=test_model_numeric)

# Confusion Matrix 4
(model_model_svm_cm <- table(test_model$Flipped, model_model_predict_svm))

# Missclassifiaction Rate
svm<- 1-sum(diag(model_model_svm_cm))/sum(model_model_svm_cm)
svm
# SVM = 0.4337349

# ROC 4
my_svm_predict <- predict(model_model_svm, newdata=test_model_numeric, type="prob")
my_svm_pred <- prediction(my_svm_predict[,1],
                          test_model_numeric$Flipped,
                          label.ordering=c(0, 1))
my_svm_perf <- performance(my_svm_pred, "tpr", "fpr")


# ROC of 4 MODELS
plot(model_lr_pref, col=1,main = "Models")
plot(model_rpart_perf, col=2, add=TRUE)
plot(model_rf_perf, col=3, add=TRUE)
plot(my_svm_perf , col=4, add=TRUE)
legend(0.7,
       0.6,
       c("Log. Reg.",
         "Class. Tree",
         "rf",
         "svm"),
       col=1:4,
       lwd=3)



#### Prediction data ####
prediction_data <- read.table("prediction_data.csv", header=TRUE, row.names=1, sep=",",
                              colClasses=c("character", "numeric",
                                           rep("factor",3), rep("numeric",7)))

# Identifying missing values in each column
summary(prediction_data)
apply(is.na(prediction_data),2,sum)
sum(is.na(prediction_data))

#Imputing missing values with median
data_med_impute <- preProcess(prediction_data, method="medianImpute")
prediction_data <- predict(data_med_impute, prediction_data)
sum(is.na(prediction_data))

# Prediction using classification tree model
My_model_rpart_predict <- predict(My_model_rpart, newdata=prediction_data, type="class")
My_model_rpart_predict
write.csv(My_model_rpart_predict,file = "predictionrpart.txt")
