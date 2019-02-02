# Title: C4T3_Galaxy_sentiment_classify_pipeline

# Last update: 2018.09.06

# File: C4T3_Galaxy_sentiment_classify_pipeline.R
# iphone_smallmatrix_labeled_8d.csv
# galaxy_smallmatrix_labeled_9d.csv
#largematrix.csv

###############
# Project Notes
###############

# Summarize project: Our analytic goal is to build models that understand the patterns in the two small matrices and then use those models with the Large 
# Matrix to predict sentiment for iPhone and Galaxy. Our next steps are as follows:
# Set up parallel processing
# Explore the Small Matrices to understand the attributes
# Preprocessing & Feature Selection
# Model Development and Evaluation
# Feature Engineering
# Apply Model to Large Matrix and get Predictions
# Analyze results, write up findings report
# Write lessons learned report

# Summarize top model and/or filtered dataset


###############
# Housekeeping
###############

# Clear objects if necessary
rm(list = ls())

# get working directory
getwd()
# set working directory
setwd("/Users/donbice/Desktop/R data files and pipeline")
dir()


################
# Load packages
################

install.packages("caret")
install.packages("corrplot")
install.packages("readr")
install.packages("doMC")    # parallel processing
library(caret)
library(corrplot)
library(doMC)
library(mlbench)
library(readr)
library(doParallel)
library(plotly)
library(e1071)
library(C50)
library(kknn)
library (dplyr)

#####################
# Parallel processing
#####################

#--- for OSX ---#
library(doMC)
detectCores()   
registerDoMC(cores = 3)


#--- for Win ---#
library(doParallel) 

## method 1
workers = makeCluster(4, type='SOCK')
registerDoParallel(workers)
foreach(i=1:4) %dopar% Sys.getpid()
# or
cluster <- makeCluster(detectCores() - 1) # avail clusters minus 1
registerDoParallel(7)  # if have 8 cores
# add in train(): doParallels = TRUE 

# method 2
# Check number of cores and workers available 
detectCores()
getDoParWorkers()
cl <- makeCluster(detectCores()-1, type='PSOCK')
registerDoParallel(cl)
# stop cluster/parallel
stopCluster(cl) 
# reactivate parallel
registerDoParallel(cl)


###############
# Import data
##############

#--- Load raw datasets ---#

## Load Train/Existing data 
galaxyDF <- read.csv("/Users/donbice/Desktop/R data files and pipeline/c4t3/galaxy_smallmatrix_labeled_9d.csv", stringsAsFactors = FALSE)
class(galaxyDF)  # "data.frame"


## Load Predict/New data ---#
# load the data
galaxy.large.matrix <- read.csv("/Users/donbice/Desktop/R data files and pipeline/c4t3/galaxyLargeMatrix.csv", stringsAsFactors = FALSE)
class(galaxyDF)  # "data.frame"



#--- Load preprocessed datasets ---#

ds_name <- read.csv("dataset_name.csv", stringsAsFactors = FALSE) 


################
# Evaluate data
################

#--- Dataset 1 ---#
str(galaxyDF) # 12911 obs. of  59 variables
summary(galaxyDF)
head(galaxyDF)
tail(galaxyDF)
names(galaxyDF)
attributes(galaxyDF)
# plot
plot_ly(galaxyDF, x= galaxyDF$galaxysentiment, type= 'histogram')

qqnorm(galaxyDF$galaxysentiment)
# check for missing values 
anyNA(galaxyDF) #FALSE
is.na(galaxyDF)


#--- Dataset 2 ---#

# If there is a dataset with unseen data to make predictions on, then preprocess here
# to make sure that it is preprocessed the same as the training dataset.
#LARGE MATRIX.csv HERE#####
str(galaxy.large.matrix) # 30418 of 60 variables
summary(galaxy.large.matrix)
head(galaxy.large.matrix)
tail(galaxy.large.matrix)
names(galaxy.large.matrix)
attributes(galaxy.large.matrix)

galaxy.large.matrix$galaxysentiment <- as.factor(galaxy.large.matrix$galaxysentiment)

galaxy.large.matrix$id <- NULL


###########################
# Feature selection/removal
###########################

#--- Examine Correlation ---#
#Use the cor() and corrplot() functions to understand the correlations with the dependent variable. Note any highly correlated features for removal. 
options(max.print = 1000)
corrData <- cor(galaxyDF)
corrData
# find attributes that are highly corrected (ideally >0.75)
highlyCorrelated <- findCorrelation(corrData, cutoff=0.75)
# print indexes of highly correlated attributes
print(highlyCorrelated)
#[1] 29 44 24 32 56 54 34 49 47 19 39 42 43 21 31 26 51 38 41 36 46 16 28 33 35 18 40 25 57 55  6  5
library(corrplot)
corrplot(corrData)

# create a new data set and remove features highly correlated with the dependent var
galaxyCOR <- galaxyDF
#iphoneCOR$featureToRemove <- NULL
galaxyCOR[,highlyCorrelated] <- NULL
str(galaxyCOR) #27 variables


#--- Examine Feature Variance ---#
#nearZeroVar() with saveMetrics = TRUE returns an object containing a table including: frequency ratio, percentage unique, zero variance and near zero variance 
nzvMetrics <- nearZeroVar(galaxyDF, saveMetrics = TRUE)
nzvMetrics

# nearZeroVar() with saveMetrics = FALSE returns an vector 
nzv <- nearZeroVar(galaxyDF, saveMetrics = FALSE) 
nzv
#[1]  3  4  6  7  9 10 11 12 13 14 15 16 17 19 20 21 22 24 25 26 27 29 30 31 32 34 35 36 37 39 40 41 42 44 45 46 47 49 50 51 52 53 54 55 56 57 58
# create a new data set and remove near zero variance features
galaxyNZV <- galaxyDF[,-nzv]
str(galaxyNZV) #12 variables

# 'data.frame':	12911 obs. of  12 variables:
#   $ iphone         : int  1 1 1 0 1 2 1 1 4 1 ...
# $ samsunggalaxy  : int  0 0 1 0 0 0 0 0 0 0 ...
# $ htcphone       : int  0 0 0 1 0 0 0 0 0 0 ...
# $ iphonecampos   : int  0 0 1 0 0 1 0 0 0 0 ...
# $ iphonecamunc   : int  0 0 0 0 0 0 0 0 0 0 ...
# $ iphonedispos   : int  0 1 0 0 0 0 2 0 0 0 ...
# $ iphonedisneg   : int  0 1 0 0 0 0 0 0 0 0 ...
# $ iphonedisunc   : int  0 1 0 0 0 0 0 0 0 0 ...
# $ iphoneperpos   : int  0 0 0 0 0 0 0 0 0 0 ...
# $ iphoneperneg   : int  0 0 0 0 0 0 0 0 0 0 ...
# $ iphoneperunc   : int  0 0 0 0 0 0 0 0 0 0 ...
# $ galaxysentiment: int  6 4 4 1 2 1 4 6 6 6 ...

#--- Recursive Feature Elimination ---#
# Let's sample the data before using RFE
set.seed(123)
galaxySample <- galaxyDF[sample(1:nrow(galaxyDF), 1000, replace=FALSE),]

# Set up rfeControl with randomforest, repeated cross validation and no updates
ctrl <- rfeControl(functions = rfFuncs, 
                   method = "repeatedcv",
                   repeats = 5,
                   verbose = FALSE)

# Use rfe and omit the response variable (attribute 59 galaxysentiment) 
rfeResults <- rfe(galaxySample[,1:58], 
                  galaxySample$galaxysentiment, 
                  sizes=(1:58), 
                  rfeControl=ctrl)

# Get results
rfeResults

# Recursive feature selection
# Outer resampling method: Cross-Validated (10 fold, repeated 5 times) 
# Resampling performance over subset size:
#   
#   Variables  RMSE Rsquared    MAE RMSESD RsquaredSD   MAESD Selected
# 1 1.528   0.3006 1.1526 0.1264    0.10129 0.07964         
# 2 1.520   0.3105 1.1741 0.1215    0.10197 0.07522         
# 3 1.520   0.3163 1.1945 0.1150    0.09971 0.07054         
# 4 1.519   0.3245 1.2053 0.1203    0.10467 0.07317         
# 5 1.510   0.3445 1.2043 0.1147    0.11140 0.07047         
# 6 1.435   0.3834 1.0732 0.1397    0.11305 0.08447         
# 7 1.417   0.4004 1.0580 0.1361    0.11290 0.07732         
# 8 1.401   0.4137 1.0458 0.1372    0.11606 0.07788         
# 9 1.383   0.4259 0.9908 0.1419    0.11191 0.08087         
# 10 1.382   0.4267 0.9955 0.1416    0.11224 0.08058        *

# Plot results
plot(rfeResults, type=c("g", "o"))

#The resulting table and plot display each subset and its accuracy and kappa. Asterisk denotes number of features judged most optimal from RFE.
#After identifying features for removal, create new dataset and add dependent variable.

# create new data set with rfe recommended features
galaxyRFE <- galaxyDF[,predictors(rfeResults)]

# add the dependent variable to galaxyRFE
galaxyRFE$galaxysentiment <- galaxyDF$galaxysentiment

# review outcome
str(galaxyRFE)


# 'data.frame':	12911 obs. of  11 variables:
#   $ iphone         : int  1 1 1 0 1 2 1 1 4 1 ...
# $ iphonedispos   : int  0 1 0 0 0 0 2 0 0 0 ...
# $ iphonedisunc   : int  0 1 0 0 0 0 0 0 0 0 ...
# $ googleandroid  : int  0 0 0 0 0 0 0 0 0 0 ...
# $ iphonecampos   : int  0 0 1 0 0 1 0 0 0 0 ...
# $ samsunggalaxy  : int  0 0 1 0 0 0 0 0 0 0 ...
# $ sonyxperia     : int  0 0 0 0 0 0 0 0 0 0 ...
# $ iphoneperpos   : int  0 0 0 0 0 0 0 0 0 0 ...
# $ htcphone       : int  0 0 0 1 0 0 0 0 0 0 ...
# $ htccampos      : int  0 0 0 0 0 0 0 0 0 0 ...
# $ galaxysentiment: int  6 4 4 1 2 1 4 6 6 6 ...

#After preprocessing have the following datasets: galaxyDF (retains all features for "out of the box" modeling), galaxyCOR, galaxyNZV, and galaxyRFE.

#############
# Preprocess
#############

#--- Dataset 1 ---#

# change data type of dependent variable in all 4 datasets!!
#DatasetName$ColumnName <- as.typeofdata(DatasetName$ColumnName
galaxyDF$galaxysentiment <- as.factor(galaxyDF$galaxysentiment)
galaxyCOR$galaxysentiment <- as.factor(galaxyCOR$galaxysentiment)
galaxyNZV$galaxysentiment <- as.factor(galaxyNZV$galaxysentiment)
galaxyRFE$galaxysentiment <- as.factor(galaxyRFE$galaxysentiment)

# rename a column
names(DatasetName)<-c("ColumnName","ColumnName","ColumnName") 
# handle missing values 
na.omit(DatasetName$ColumnName)
na.exclude(DatasetName$ColumnName)        
DatasetName$ColumnName[is.na(DatasetName$ColumnName)] <- mean(DatasetName$ColumnName,na.rm = TRUE)
# discretize (if applicable)


#--- Dataset 2 ---#






################
# Sampling
################

#Per plan of attack, no sampling.

#######################OUT OF BOX iphoneDF##############################################

##################
# Train/test sets
##################

# create the training partition that is 70% of total obs
set.seed(123) # set random seed
inTraining <- createDataPartition(galaxyDF$galaxysentiment, p=0.70, list=FALSE)
# create training/testing dataset
trainSet <- galaxyDF[inTraining,]   
testSet <- galaxyDF[-inTraining,]   
# verify number of obs 
str(trainSet)  
str(testSet)   


################
# Train control
################

# set 10 fold cross validation
fitControl <- trainControl(method = "repeatedcv", number = 10, repeats = 5)


##############
# Train model
##############


## ------- C5.0 ------- ##

# C50 train/fit
set.seed(123)
c50Fit1 <- train(galaxysentiment~., data=trainSet, method="C5.0", importance=T, trControl=fitControl) #importance is needed for varImp
c50Fit1




## ------- RF ------- ##

# RF train/fit
set.seed(123)
rfFit1 <- train(galaxysentiment~., data=trainSet, method="rf", importance=T, trControl=fitControl)
rfFit1



## ------- SVM ------- ##

# SVM train/fit
set.seed(123)
svmFit1 <- train(galaxysentiment~., data=trainSet, method="svmLinear", trControl=fitControl)
svmFit1



## ------- kkNN ------- ##

# kkNN train/fit
set.seed(123)
kkNNFit1 <- train(galaxysentiment~., data=trainSet, method="kknn", trControl=fitControl)
kkNNFit1



#################
# Evaluate models
#################

##--- Compare models ---##

# use resamples to compare model performance
ModelFitResults <- resamples(list(c50=c50Fit1, rf=rfFit1, svm=svmFit1, kknn=kkNNFit1))
# output summary metrics for tuned models 
summary(ModelFitResults)
# ds (galaxyDF) Make a note of the dataset that the performance metrics belong to.
# Note performance metrics. Add the summary output as a comment.

# summary.resamples(object = ModelFitResults)
# 
# Models: c50, rf, svm, kknn 
# Number of resamples: 50 
# 
# Accuracy 
# Min.   1st Qu.    Median      Mean   3rd Qu.      Max. NA's
# c50  0.7436464 0.7587159 0.7634051 0.7642734 0.7717213 0.7964602    0
# rf   0.7433628 0.7585825 0.7648038 0.7641190 0.7699115 0.7975664    0
# svm  0.6913717 0.7062726 0.7145225 0.7142921 0.7224693 0.7378319    0
# kknn 0.6920530 0.7302375 0.7401885 0.7393860 0.7497919 0.7643805    0
# 
# Kappa 
#           Min.   1st Qu.    Median      Mean   3rd Qu.      Max. NA's
# c50  0.4788794 0.5090293 0.5286761 0.5267322 0.5416820 0.6036618    0
# rf   0.4847004 0.5158268 0.5331368 0.5306662 0.5455967 0.6088431    0
# svm  0.3530077 0.3833582 0.3996677 0.4009025 0.4184005 0.4642260    0
# kknn 0.4377875 0.4734727 0.4955479 0.4947047 0.5128376 0.5547889    0
# 

########################
# Predict with top model(s)
########################

# make predictions
c50Pred1 <- predict(c50Fit1, testSet)
#performace measurement
postResample(c50Pred1, testSet$galaxysentiment)

# (make  note of performance metrics)
#plot predicted verses actual
plot(c50Pred1,testSet$galaxysentiment)
# print predictions
c50Pred1

# Create a confusion matrix from C50 predictions 
cmC50 <- confusionMatrix(c50Pred1, testSet$galaxysentiment) 
cmC50


# make predictions
rfPred1 <- predict(rfFit1, testSet)
#performace measurement
postResample(rfPred1, testSet$galaxysentiment)

# (make  note of performance metrics)
#plot predicted verses actual
plot(rfPred1,testSet$galaxysentiment)
# print predictions
rfPred1

# Create a confusion matrix from random forest predictions 
cmRF <- confusionMatrix(rfPred1, testSet$galaxysentiment) 
cmRF






####################### galaxy COR ##############################################

##################
# Train/test sets
##################

# create the training partition that is 70% of total obs
set.seed(123) # set random seed
inTrainingCOR <- createDataPartition(galaxyCOR$iphonesentiment, p=0.70, list=FALSE)
# create training/testing dataset
trainSetCOR <- galaxyCOR[inTrainingCOR,]   
testSetCOR <- galaxyCOR[-inTrainingCOR,]   
# verify number of obs 
str(trainSetCOR)  
str(testSetCOR)   


################
# Train control
################

# set 10 fold cross validation
fitControlCOR <- trainControl(method = "repeatedcv", number = 10, repeats = 5)


##############
# Train model
##############

## ------- C5.0 ------- ##

# C50 train/fit
set.seed(123)
c50FitCOR <- train(galaxysentiment~., data=trainSetCOR, method="C5.0", importance=T, trControl=fitControl) 
c50FitCOR


## ------- RF ------- ##

# RF train/fit
set.seed(123)
rfFitCOR <- train(galaxysentiment~., data=trainSetCOR, method="rf", importance=T, trControl=fitControl) 
rfFitCOR


#################
# Evaluate models
#################

##--- Compare models ---##

# use resamples to compare model performance
ModelFitResultsCOR <- resamples(list(c50=c50FitCOR, rf=rfFitCOR))
# output summary metrics for tuned models 
summary(ModelFitResultsCOR)
# ds (iphoneCOR) Make a note of the dataset that the performance metrics belong to.
# Note performance metrics. Add the summary output as a comment.



######################  iphoneNZV ##############################################

##################
# Train/test sets
##################

# create the training partition that is 70% of total obs
set.seed(123) # set random seed
inTrainingNZV <- createDataPartition(galaxyNZV$galaxysentiment, p=0.70, list=FALSE)
# create training/testing dataset
trainSetNZV <- galaxyNZV[inTrainingNZV,]   
testSetNZV <- galaxyNZV[-inTrainingNZV,]   
# verify number of obs 
str(trainSetNZV)  
str(testSetNZV)   


################
# Train control
################

# set 10 fold cross validation
fitControlNZV <- trainControl(method = "repeatedcv", number = 10, repeats = 5)


##############
# Train model
##############


## ------- C5.0 ------- ##

# C50 train/fit
set.seed(123)
c50FitNZV <- train(galaxysentiment~., data=trainSetNZV, method="C5.0", importance=T, trControl=fitControl) 
c50FitNZV

# model  winnow  trials  Accuracy   Kappa    
# rules  FALSE    1      0.7501808  0.4916002
# rules  FALSE   10      0.7399152  0.4672273
# rules  FALSE   20      0.7399152  0.4672273
# rules   TRUE    1      0.7494949  0.4906252
# rules   TRUE   10      0.7389180  0.4646270
# rules   TRUE   20      0.7389180  0.4646270
# tree   FALSE    1      0.7496941  0.4913023
# tree   FALSE   10      0.7393621  0.4694748
# tree   FALSE   20      0.7393621  0.4694748
# tree    TRUE    1      0.7493623  0.4910048
# tree    TRUE   10      0.7365315  0.4617824
# tree    TRUE   20      0.7365315  0.4617824
# 
# Accuracy was used to select the optimal model using the largest value.
# The final values used for the model were trials = 1, model = rules and winnow = FALSE.



## ------- RF ------- ##

# RF train/fit
set.seed(123)
rfFitNZV <- train(galaxysentiment~., data=trainSetNZV, method="rf", importance=T, trControl=fitControl) 
rfFitNZV



#################
# Evaluate models
#################

##--- Compare models ---##

# use resamples to compare model performance
ModelFitResultsNZV <- resamples(list(c50=c50FitNZV, rf=rfFitNZV))
# output summary metrics for tuned models 
summary(ModelFitResultsNZV)
# ds (iphoneNZV) Make a note of the dataset that the performance metrics belong to.
# Note performance metrics. Add the summary output as a comment.



##############################  galaxy RFE #######################################

##################
# Train/test sets
##################

# create the training partition that is 70% of total obs
set.seed(123) # set random seed
inTrainingRFE <- createDataPartition(galaxyRFE$galaxysentiment, p=0.70, list=FALSE)
# create training/testing dataset
trainSetRFE <- galaxyRFE[inTrainingRFE,]   
testSetRFE <- galaxyRFE[-inTrainingRFE,]   
# verify number of obs 
str(trainSetRFE)  
str(testSetRFE)   


################
# Train control
################

# set 10 fold cross validation
fitControlRFE <- trainControl(method = "repeatedcv", number = 10, repeats = 5)


##############
# Train model
##############


## ------- C5.0 ------- ##

# C50 train/fit
set.seed(123)
c50FitRFE <- train(galaxysentiment~., data=trainSetRFE, method="C5.0", importance=T, trControl=fitControl) 
c50FitRFE

# model  winnow  trials  Accuracy   Kappa    
# rules  FALSE    1      0.7515076  0.5000440
# rules  FALSE   10      0.7442976  0.4833589
# rules  FALSE   20      0.7442976  0.4833589
# rules   TRUE    1      0.7516182  0.4994803
# rules   TRUE   10      0.7437431  0.4803712
# rules   TRUE   20      0.7437431  0.4803712
# tree   FALSE    1      0.7509986  0.5033764
# tree   FALSE   10      0.7433662  0.4821249
# tree   FALSE   20      0.7433662  0.4821249
# tree    TRUE    1      0.7513527  0.5027389
# tree    TRUE   10      0.7425477  0.4785152
# tree    TRUE   20      0.7425477  0.4785152
# 
# Accuracy was used to select the optimal model using the largest value.
# The final values used for the model were trials = 1, model = rules and winnow = TRUE.

## ------- RF ------- ##

# RF train/fit
set.seed(123)
rfFitRFE <- train(galaxysentiment~., data=trainSetRFE, method="rf", importance=T, trControl=fitControl) #importance is needed for varImp
rfFitRFE



#################
# Evaluate models
#################

##--- Compare models ---##

# use resamples to compare model performance
ModelFitResults <- resamples(list(c50=c50FitRFE, rf=rfFitRFE))
# output summary metrics for tuned models 
summary(ModelFitResults)
# ds (iphoneRFE) Make a note of the dataset that the performance metrics belong to.
# Note performance metrics. Add the summary output as a comment.





######################################################################################################

##--- Conclusion ---##
# Note which model is top model, and why.





##--- Save top performing model ---##

# save model 
saveRDS()  # Q: What type of object does saveRDS create?
# load and name model to make predictions with new data
LMfit1 <- readRDS() # Q: What type of object does readRDS create?




########################
# Feature Engineering
########################

#--- Engineering the dependent variable by using dplyr package recode() function to combine sentiment levels 1 negative, 2 somewhat neg, 3 somewhat pos, 4 positive  ---#
# create a new dataset that will be used for recoding sentiment
galaxyRC <- galaxyDF
# recode sentiment to combine factor levels 1 & 2 and 5 & 6
galaxyRC$galaxysentiment <- recode(galaxyRC$galaxysentiment, '1' = 1, '2' = 1, '3' = 2, '4' = 3, '5' = 4, '6' = 4) 
# inspect results
summary(galaxyRC)
str(galaxyRC) 
# make iphonesentiment a factor
galaxyRC$galaxysentiment <- as.factor(galaxyRC$galaxysentiment)


#Model best learner and note any improvement in accuracy and kappa

# Train/test sets

# create the training partition that is 70% of total obs
set.seed(123) # set random seed
inTrainingRC <- createDataPartition(galaxyRC$galaxysentiment, p=0.70, list=FALSE)
# create training/testing dataset
trainSetRC <- galaxyRC[inTrainingRC,]   
testSetRC <- galaxyRC[-inTrainingRC,]   
# verify number of obs 
str(trainSetRC)  
str(testSetRC)   


# Train control
# set 10 fold cross validation
fitControlRC <- trainControl(method = "repeatedcv", number = 10, repeats = 3)

# Train model
## ------- C5.0 ------- ##

# C50 train/fit
set.seed(123)
c50FitRC <- train(galaxysentiment~., data=trainSetRC, method="C5.0", importance=T, trControl=fitControl) 
c50FitRC

# C5.0 
# 9039 samples
# 58 predictor
# 4 classes: '1', '2', '3', '4' 
# No pre-processing
# Resampling: Cross-Validated (10 fold, repeated 5 times) 
# Summary of sample sizes: 8136, 8134, 8136, 8136, 8134, 8133, ... 
# Resampling results across tuning parameters:
#   
#   model  winnow  trials  Accuracy   Kappa    
# rules  FALSE    1      0.8443196  0.5956194
# rules  FALSE   10      0.8400503  0.5853437
# rules  FALSE   20      0.8400503  0.5853437
# rules   TRUE    1      0.8437670  0.5939304
# rules   TRUE   10      0.8374389  0.5781437
# rules   TRUE   20      0.8374389  0.5781437
# tree   FALSE    1      0.8435455  0.5950720
# tree   FALSE   10      0.8384769  0.5809058
# tree   FALSE   20      0.8384769  0.5809058
# tree    TRUE    1      0.8435234  0.5949702
# tree    TRUE   10      0.8356449  0.5753931
# tree    TRUE   20      0.8356449  0.5753931
# 
# Accuracy was used to select the optimal model using the largest value.
# The final values used for the model were trials = 1, model = rules and winnow = FALSE.





#---Principal Component Analysis using caret preProcess() function---#

# data = training and testing from iphoneDF (no feature selection) 
# create object containing centered, scaled PCA components from training set
# excluded the dependent variable and set threshold to .95
preprocessParams <- preProcess(trainSet[,-59], method=c("center", "scale", "pca"), thresh = 0.95)
print(preprocessParams)

#Components needed to capture 95% variance
#PCA needed 25 components to capture 95 percent of the variance

# use predict to apply pca parameters, create training, exclude dependent
train.pca <- predict(preprocessParams, trainSet[,-59])

# add the dependent to training
train.pca$galaxysentiment <- trainSet$galaxysentiment

# use predict to apply pca parameters, create testing, exclude dependent
test.pca <- predict(preprocessParams, testSet[,-59])

# add the dependent to training
test.pca$galaxysentiment <- testSet$galaxysentiment

# use predict to apply pca parameters, create training, exclude dependent
train.pca <- predict(preprocessParams, trainSet[,-59])

# add the dependent to training
train.pca$galaxysentiment <- trainSet$galaxysentiment

# use predict to apply pca parameters, create testing, exclude dependent
test.pca <- predict(preprocessParams, testSet[,-59])

# add the dependent to training
test.pca$galaxysentiment <- testSet$galaxysentiment

# inspect results
str(train.pca)
str(test.pca)

#Model using best learner. Note that trainSet and testSet in original replace by train.pca and test.pca. Did accuracy and kappa increase? Did you review the confusion matrix? 

## ------- C5.0 ------- ##

# C50 train/fit
set.seed(123)
c50FitPCA <- train(galaxysentiment~., data=train.pca, method="C5.0", importance=T, trControl=fitControl)
c50FitPCA

# model  winnow  trials  Accuracy   Kappa    
# rules  FALSE    1      0.7521070  0.5005491
# rules  FALSE   10      0.7499366  0.4976065
# rules  FALSE   20      0.7499366  0.4976065
# rules   TRUE    1      0.7534786  0.5033774
# rules   TRUE   10      0.7490081  0.4960454
# rules   TRUE   20      0.7490081  0.4960454
# tree   FALSE    1      0.7544282  0.5074113
# tree   FALSE   10      0.7501357  0.4991887
# tree   FALSE   20      0.7501357  0.4991887
# tree    TRUE    1      0.7539428  0.5059435
# tree    TRUE   10      0.7490963  0.4969857
# tree    TRUE   20      0.7490963  0.4969857
# 
# Accuracy was used to select the optimal model using the largest value.
# The final values used for the model were trials = 1, model = tree and winnow = FALSE.

## ------- RF ------- ##

# RF train/fit
set.seed(123)
rfFitPCA <- train(galaxysentiment~., data=train.pca, method="rf", importance=T, trControl=fitControl)
rfFitPCA

##--- Compare models ---##

# use resamples to compare model performance
ModelFitResultsPCA <- resamples(list(c50=c50FitPCA, rf=rfFitPCA))
# output summary metrics for tuned models 
summary(ModelFitResultsPCA)
# ds (galaxyPCA) Make a note of the dataset that the performance metrics belong to.
# Note performance metrics. Add the summary output as a comment.



##--- Compare models ---##

# use resamples to compare model performance across Feature selection and feature engineering versus out of box
ModelFitResultsFinal <- resamples(list(c50Fit1,c50FitNZV, c50FitRFE, c50FitRC, c50FitPCA))
# output summary metrics for tuned models 
summary(ModelFitResultsFinal)
# ds (galaxyPCA) Make a note of the dataset that the performance metrics belong to.
# Note performance metrics. Add the summary output as a comment.

# Call:
#   summary.resamples(object = ModelFitResultsFinal)
# 
# Models: Model1, Model2, Model3, Model4, Model5 
# Number of resamples: 50 
# 
# Accuracy 
# Min.   1st Qu.    Median      Mean   3rd Qu.      Max. NA's
# Model1 0.7436464 0.7587159 0.7634051 0.7642734 0.7717213 0.7964602    0
# Model2 0.7278761 0.7433628 0.7504153 0.7501808 0.7565023 0.7776549    0
# Model3 0.7300885 0.7453795 0.7518005 0.7516182 0.7571982 0.7787611    0
# Model4 0.8185841 0.8372543 0.8445788 0.8443196 0.8517289 0.8615725    0
# Model5 0.7303867 0.7473007 0.7548425 0.7544282 0.7612473 0.7865044    0
# 
# Kappa 
# Min.   1st Qu.    Median      Mean   3rd Qu.      Max. NA's
# Model1 0.4788794 0.5090293 0.5286761 0.5267322 0.5416820 0.6036618    0
# Model2 0.4348119 0.4724540 0.4951014 0.4916002 0.5058839 0.5567698    0
# Model3 0.4522842 0.4853200 0.4965827 0.4994803 0.5145697 0.5648260    0
# Model4 0.5237704 0.5783597 0.5998281 0.5956194 0.6155807 0.6460633    0
# Model5 0.4555586 0.4940139 0.5072808 0.5074113 0.5199603 0.5827646    0


########################
# Predict with top model
########################

# make predictions
c50PredFinal <- predict(c50FitRC, galaxy.large.matrix)

# print predictions
c50PredFinal

summary(c50PredFinal)

#All features galaxyDF
# 0     1     2     3     4     5 
# 11633     0   677  1768   484 15856 

#Recode to 4 classes galaxyRC
#> summary(c50PredFinal)
# 1     2     3     4 
# 11556   674  1789 16399 

