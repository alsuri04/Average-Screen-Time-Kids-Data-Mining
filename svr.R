# SVR

# Importing the dataset
dataset = read.csv('Student_Performance.csv')
# Fitting SVR to the whole dataset
head(dataset)

library(e1071)
library(caTools)
dataset$Extracurricular.Activities = factor(dataset$Extracurricular.Activities,
                        levels = c('No', 'Yes'),
                        labels = c(0,1))

split <- sample.split(dataset$  Performance.Index, SplitRatio = 0.75)
training_set <- subset(dataset, split == TRUE)
test_set <- subset(dataset, split == FALSE)
head(dataset)
regressor = svm(formula = Performance.Index ~ .,
                data = training_set,
                type = 'eps-regression',
                kernel = 'radial')

# Making a  prediction
y_pred <-predict(regressor,newdata=test_set)
print(y_pred)

ssr = sum((test_set$Performance.Index - y_pred) ^ 2)
sst = sum((test_set$Performance.Index - mean(test_set$Performance.Index)) ^ 2)
r2 = 1 - (ssr/sst)
print(r2)
r2_adjusted = 1 - (1 - r2) * (length(test_set$Performance.Index) - 1) / (length(test_set$Performance.Index) - 6 - 1)
print(paste('adjusted svr', r2_adjusted))
library(ggplot2)
x_range <- range(dataset$Previous.Scores)
y_range <- range(dataset$Performance.Index)
print(ggplot() +
        geom_point(aes(x = test_set$Previous.Scores, y = test_set$Performance.Index), colour = 'red') +
        geom_point(aes(x = test_set$Previous.Scores, y = predict(regressor, newdata = test_set)), colour = 'blue') +
        ggtitle('Previous score vs Current Performance Index ') +
        xlab('Previous Score') + ylab('Current Performance Index') +
        xlim(x_range) + ylim(y_range))    
