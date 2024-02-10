##################################################################
#
# Author: Laurence Burden
# IN400 - AI: Deep Learning and Machine Learning
# Unit 3 Assignment / Modeule 2 Competency Assessment Part 2
# Machine Learning Classification Using the Iris Dataset with R
#
##################################################################


# Packages used
packages <- c("tidyr", "ggplot2", "rpart", "rpart.plot", "glue")

# Install needed packages
installed_packages <- packages %in% rownames(installed.packages())

if (any(installed_packages == FALSE)) {
  install.packages(packages[!installed_packages])
}

# Load packages
invisible(lapply(packages, library, character.only = TRUE))

# Import libraries
library("tidyr")
library("ggplot2")
library("rpart")
library("rpart.plot")

# Load dataset
data(iris)
iris_dataset <- iris

# Explore data by peeking at it and its structure
head(iris_dataset)
dim(iris_dataset)
str(iris_dataset)
summary(iris_dataset)

# Assign meaningful column names based on dataset repository
colnames(iris_dataset) <- c("Sepal.Length",
                            "Sepal.Width",
                            "Petal.Length",
                            "Petal.Width",
                            "Species")
head(iris_dataset)

# Check classification number for flowers
levels(iris_dataset)

# Visualizations
# Scatter Plot
sp <- ggplot(data = iris_dataset,
             aes(x = Petal.Length, y = Petal.Width)) +
             geom_point(aes(color = Species, shape = Species)) +
             xlab("Petal Length") +
             ylab("Petal Width") +
             ggtitle("Petal Length-Width") +
             geom_smooth(method="lm")
print(sp)

# Histogram
histogram <- ggplot(iris_dataset,
                 aes(x = Sepal.Width)) +
  geom_histogram(binwidth = 0.2, color = "black",
                 aes(fill = Species)) +
  xlab("Sepal Width") +
  ylab("Frequency") +
  ggtitle("Histogram of Sepal Width")

print(histogram)

# Box plot
box <- ggplot(data = iris_dataset,
              aes(x = Species, y = Sepal.Length)) +
  geom_boxplot(aes(fill = Species)) +
  ylab("Sepal Length") +
  ggtitle("Iris Boxplot") +
  stat_summary(fun.y = mean, geom = "point", shape = 5, size = 4)
  
print(box)

# Classification
# First split data in 80-20 train/test data
n <- seq_len(nrow(iris_dataset))
index <- sample(n, length(n) * 0.8)
trainset <- iris_dataset[index,]
testset <- iris_dataset[-index,]

# Peek at the training data
head(trainset)
dim(trainset)
str(trainset)

# Peek at test data
head(testset)
dim(testset)
str(testset)

# Build classification model using a decision tree classifier
# First, fit the data
fit <- rpart(Species~., data = trainset, method = 'class')
print(rpart.plot(fit))

# Use the fit to perform predictions on the training set
pred_train <- predict(object = fit, newdata = trainset[,1:4], type = "class")
print(pred_train)

# Metrics
# Determine accuracy metrics with a confusion matrix
print(table(pred_train, trainset$Species))

# Use the fit to perform predictions on the test data
pred_test <- predict(object = fit, newdata = testset[,1:4], type = "class")
print(pred_test)

# Confusion matrix on test data
print(table(pred_test, testset$Species))