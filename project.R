library(caret)

# remove from the training set "", "NA" and "#DIV/0!" observations:
training<- read.csv("~/Desktop/.../pml-training.csv", na.strings=c("","NA","#DIV/0!"))
testing<- read.csv("~/Desktop/.../pml-testing.csv")

# select useful variables in both sets:
user<-grep("user",names(training), value=FALSE) 
belt<-grep("belt",names(training), value=FALSE) 
arm<-grep("[^fore]arm",names(training), value=FALSE)
forearm<-grep("forearm",names(training), value=FALSE)
dumbbell<-grep("dumbbell",names(training), value=FALSE)
classe<-grep("classe",names(training), value=FALSE)

training<-training[,c(user,belt, arm, forearm, dumbbell, classe)]
testing<-testing[,c(user,belt, arm, forearm, dumbbell, classe)]

# remove variables containing just NA values:
training<-training[,apply(apply(training,2,is.na),2,sum)==0]
testing<-testing[,apply(apply(testing,2,is.na),2,sum)==0]

# Data splitting from the training set:
inTrain<-createDataPartition(y=training$classe, p=0.7, list=FALSE)
train<-training[inTrain,]
test<-training[-inTrain,]

# --
# Building some models: obtaining the accuracy.
# The best accuracy results are obtained with Random Forest algorithm (Accuracy : 0.9925).
# --

mod1 <- train(classe ~.,method="lda",data=train)
pred1<-predict(mod1,test)
confusionMatrix(pred1, test$classe) # Accuracy : 0.7276 

mod2 <- train(classe ~.,method="svmLinear",data=train) # support vector machine
pred2<-predict(mod2,test)
confusionMatrix(pred2, test$classe) # Accuracy : 0.8034

mod3 <- train(classe ~.,method="gbm",data=train) # boosting
pred3<-predict(mod3,test)
confusionMatrix(pred3, test$classe) # Accuracy : 0.9601 

mod4 <- train(classe ~.,method="pcaNNet",data=train) # Neural Networks with Feature Extraction
pred4<-predict(mod4,test)
confusionMatrix(pred4, test$classe) # Accuracy : 0.6348

mod5 <- train(classe ~.,method="rpart",data=train) # model based tree
pred5<-predict(mod5,test)
confusionMatrix(pred5, test$classe) # Accuracy : 0.4748 

mod6 <- train(classe ~.,method="rf",data=train, trControl=trainControl(method="cv"), number=3) # randomforest
pred6<-predict(mod6,test)
confusionMatrix(pred6, test$classe) # Accuracy : 0.9925  

# --
# Combining some models to try increasy the accuracy.
# The combination of model does not improve the accuracy.
# --

predDF <- data.frame(pred1,pred2,pred3,pred4,pred5,pred6, classe=test$classe)
combModFit <- train(classe ~.,method="gbm",data=predDF) # the new model
combPred <- predict(combModFit,newdata=predDF)
confusionMatrix(combPred, test$classe) # Accuracy : 0.9925 

# Finally, Random Forest model is used to predict on testing set.

predict(mod6,testing) 

# The prediction results are:
# > predict(mod6,testing) 
# [1] B A B A A E D B A A B C B A E E A B B B
# Levels: A B C D E


