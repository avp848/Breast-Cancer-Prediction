library(caret)
library(corrplot)
library(ggplot2)
library(kknn)
library(rpart)
library(e1071)
data = read.csv("cData.csv")

#str(), summary(), dim(), unique values for diagnosis
#check for N/A
#we observe several unclassified data points - no diagnosis
data = data[,-c(1, 2, 34)]

#data = data[, -c("id", "y", "unnamed")]

data = data[complete.cases(data),]

###PLOT ONE VARIABLE GRAPH  FOR ALL VARIABLES
#plot diagnosis on single variable graph and identify any odd behavior
ggplot(data, aes(diagnosis)) + geom_bar(fill = "purple")

#one data point with no diagnosis code so we remove that row
data = data[-which(data$diagnosis==""),]

#how many are benign and malignant
table(data$diagnosis) #unique split
prop.table(table(data$diagnosis)) #percentage split


table(data$diagnosis)

#create a correlation plot without diagnosis
all = cor(data[,-1]) #remove first which isn't numeric
corrplot(all, method="pie", type = "lower")

data$diagnosis=as.numeric(ifelse(data$diagnosis=="M",1,0))
#create a correlation plot with diagnosis variable which is numeric
all = cor(data) 
corrplot(all)

#3 diff categories of datasets in cancer data for mean, se, and worst
#Mean(3-13), Standard Error (), and ()
#Mean
x = cor(data[,c(1:11)])
corrplot(x)

#Se
y = cor(data[,c(12:21)])
corrplot(y)

#Worst
z = cor(data[,c(21:30)])
corrplot(z)

#heatmap
col = colorRampPalette(c("blue", "white", "red"))(20)
heatmap(all, col=col, symm=F)

#analysis

##converting the diagnosis column back to factor so that the color dimension can be used
data$diagnosis = as.factor(data$diagnosis)
##plots show that in general malignant diagnoses have higher scores among the variables
p1 = ggplot(data=data, aes(y=radius_mean, fill=diagnosis)) + geom_boxplot()
#plot boxplots for all variables for mean and assign them to a variable
p2 = ggplot(data=data, aes(y=texture_mean, fill=diagnosis)) + geom_boxplot()
p3 = ggplot(data=data, aes(y=perimeter_mean, fill=diagnosis)) + geom_boxplot()
p4 = ggplot(data=data, aes(y=area_mean, fill=diagnosis)) + geom_boxplot()
p5 = ggplot(data=data, aes(y=smoothness_mean, fill=diagnosis)) + geom_boxplot()
p6 = ggplot(data=data, aes(y=compactness_mean, fill=diagnosis)) + geom_boxplot()
p7 = ggplot(data=data, aes(y=concavity_mean, fill=diagnosis)) + geom_boxplot()
p8 = ggplot(data=data, aes(y=concave.points_mean, fill=diagnosis)) + geom_boxplot()
p9 = ggplot(data=data, aes(y=symmetry_mean, fill=diagnosis)) + geom_boxplot()
p10 = ggplot(data=data, aes(y=fractal_dimension_mean, fill=diagnosis)) + geom_boxplot()

install.packages(cowplot)
library(cowplot)
plot_grid(p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, nrow = 3)


#-------------- Visualization Complete --- 
  ############## building models ############
#Split Test and Training Dataset
set.seed(123)
table(data$diagnosis)
index = createDataPartition(data$diagnosis, p=0.7, list=F)
trainingDataset = data[index,]
testingDataset = data[-index,]

##Logistic Regression
#for glm select only one category of data else it is difficult to converge 
trainingDataset_mean = trainingDataset[,c(1:11)]

#diagnosis should be numeric for logistic regression
# picking the best features to predict diagnosis of cancer. Removing collinearity by only picking one feature from heavily correlated features
#collinearity: eg: picking only one feature, eg: perimeter_mean between (area_mean, radius_mean, perimeter_mean)

prediction_features = c("diagnosis", "texture_mean", "perimeter_mean", "smoothness_mean", "compactness_mean")

#pick above features
#trainingDataset$diagnosis = as.numeric(trainingDataset$diagnosis)
trainingDataset_mean = trainingDataset_mean[, prediction_features]
model.glm = glm(diagnosis~., trainingDataset_mean, family=binomial(link="logit"))
predict.glm = predict(model.glm, testingDataset, type="response")
results = ifelse(predict.glm >= 0.5, 1, 0)
accuracy.glm = mean(results == testingDataset$diagnosis)
accuracy.glm

### CONFUSION MATRIX
cm_lg  = confusionMatrix(as.factor(results), as.factor(testingDataset$diagnosis))   
cm_lg

# KKNN model
model.kknn = train(diagnosis~., trainingDataset, method="kknn")
predict.kknn=predict(model.kknn, testingDataset)
accuracy.kknn = mean(predict.kknn == testingDataset$diagnosis)
accuracy.kknn  #[1] 0.9588235


############Rpart (recursive partitioning and regression trees)
# Run the model on entire dataset, and not just selected features to check for accuracy and gradually remove one feature at a time
model.rpart = train(diagnosis~., trainingDataset, method="rpart")
predict.rpart = predict(model.rpart, testingDataset)
accuracy.rpart = mean(predict.rpart==testingDataset$diagnosis)
accuracy.rpart #0.9117647
cm_rpart  <- confusionMatrix(predict.rpart, as.factor(testingDataset$diagnosis))   
cm_rpart

####################################### Support Vector Machines 
model.svm = svm(diagnosis~., data=trainingDataset)
predict.svm=predict(model.svm, testingDataset)
accuracy.svm = mean(predict.svm==testingDataset$diagnosis)
accuracy.svm
cm_svm=confusionMatrix(as.factor(predict.svm), as.factor(testingDataset$diagnosis))  
cm_svm

################################## Class imbalance
table(trainingDataset$diagnosis)
#There is approximately 60/40 split between benign and malignant samples
#So lets downsample it using the downSample function from caret package.
#To do this you just need to provide the X and Y variables as arguments.

# Down Sample
set.seed(100)
down_train <- downSample(x = trainingDataset[, -1], y = trainingDataset$diagnosis, yname="diagnosis")
table(down_train$diagnosis)

### new prediction to check accuracy with balanced data for knn
#down_train$diagnosis = as.factor(down_train$diagnosis)
#table(down_train$diagnosis)

model = train(diagnosis~., down_train, method="kknn")
predict = predict(model, testingDataset)
accuracy.kknn.balanced = mean(predict == testingDataset$diagnosis)
accuracy.kknn.balanced #0.9588235

#Building observation table with all the models
#Final Observations Table
Models = c("GLM","KKNN", "KKNN-B", "RPART", "SVM")
Accuracy = c(accuracy.glm, accuracy.kknn, accuracy.kknn.balanced, accuracy.rpart, accuracy.svm)
finalTable=data.frame(Models, Accuracy)


