rm(list = ls())

### Exercice 2 ###
table = iris

# Question 1
X = table[, 1:4]
Y = as.character (table[, 5])
Y[Y != "versicolor"] <- "autres"
Y = as.factor(Y)

# Question 2
set.seed(1234)
perm = sample(1:150)
X = X[perm, ]
Y = Y[perm]

Xtrain = X[1:75, ]
Ytrain = Y[1:75]
Xtest = X[76:150, ]
Ytest = Y[76:150]

# Questions 3 et 4
library(tree)
#?tree
arbre = tree(formula = Ytrain ~., data = Xtrain)
summary(arbre)
plot(arbre)
text(arbre)

# Question 5
Ypred = predict(arbre, newdata = Xtest, type ="class")
mean(Ypred != Ytest)

#iris_mod <- table
#iris_mod$Species_fus[iris_mod$Species != "versicolor"] <- "Autres"
#iris_mod$Species_fus[iris_mod$Species == "versicolor"] <- "versicolor"
