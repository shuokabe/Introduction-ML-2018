rm(list = ls())

# Exercice 1
setwd("/Users/okabeshu/Documents/ENSAE/Wine")
#1
red_wine <- read.table("winequality-red.csv", sep = ";", header = T)

#2
help(prcomp)
red_acp <- prcomp(red_wine, scale = T) 
plot(red_acp)
variances <- red_acp$sdev^2

pct_var = sum(variances[1:2] /12)
biplot(red_acp, xlabs = rep("", nrow(red_wine)))

# Exercice 2
set.seed(1234)
setwd("/Users/okabeshu/Documents/ENSAE/Movie Lens")
# 1
ratings <- read.table("ratings.dat", sep = ":", header = F, colClasses = c(NA, "NULL"))
names(ratings) <- c("user", "movie", "grade", "useless")
ratings <- ratings[sample(1:nrow(ratings)), ] # Mélange les lignes aléatoirement
train <- ratings[1:750000, ]
test <- ratings[750001:nrow(ratings), ]

# 2
help(aggregate)
mean_ratings <- aggregate(train[, 3], list(train[, 2]), mean)
preds <- round(mean_ratings[test[, 2], 2])
mean((preds - test[, 3])^2, na.rm = T) 
mean(is.na(preds))

# 3
library(softImpute)
A <- Incomplete(i = train[, 1], j = train[, 2], x = train[, 3])
MSE <- rep(NA, 10)
for (k in (1:10)*2) {
  B <- softImpute(A, rank.max = k, maxit = 500)
  pred <- impute(object = B, i = test[, 1], j = test[, 2])
  pred <- round(pmax(pmin(pred, 5), 1))
  MSE[k / 2] <- mean((pred - test[, 3])^2)
}

plot((1:10) * 2, MSE)

