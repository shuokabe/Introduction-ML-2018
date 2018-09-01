rm(list = ls())
library(plyr)

setwd("/Users/okabeshu/Documents/ENSAE/Introduction au Machine Learning/Tests fonction du cours") # Path of the main folder

##### Classification #####
### Discrimination linéaire ###
seeds <- read.table("seeds_dataset.txt")
seeds <- rename(seeds, c("V1" = "Area", "V2" = "Perim.", "V3" = "Compac.", "V4" = "lenght", "V5" = "width",
                "V6" = "asym.", "V7" = "grov.", "V8" = "variety")) # Rename columns 

# Tableaux par variables
pairs(seeds[1:7], col = seeds$variety)

# Test de prédiction avec les 2ème et 3ème colonnes uniquement
B <- seeds[c(2, 3)]
Y <- seeds$variety
Y[Y == 1] = 0
Y[Y == 2] = 1 # Rosa
Y[Y == 3] = 0

# LDA (Linear Discriminant Analysis)
library(MASS) # Modern Applied Statistics with S
resultat <- lda(Y~., data = B)
resultat

plot(resultat)

# Prédiction pour une nouvelle graine
test <- data.frame(Perim. = 14.84,Compac. = 0.871)
predict(resultat, newdata = test)

# Matrice de confusion
Ypred <- predict(resultat, newdata = B)$class
table(Y, Ypred)

### Règles plug-in ###
A <- iris[sample(1:150), ]
pairs(A[1:4], col = A$Species)

# Bases d'apprentissage et de test
Xtrain = A[1:100,1:4]
Ytrain = A[1:100,5]
Xtest = A[101:150,1:4]
Ytest = A[101:150,5]

# Prédiction
library(class)
Ypred <- knn(train = Xtrain, test = Xtest, cl = Ytrain, k = 3) # k-plus proches voisins
table(Ypred,Ytest)

### Sélection de prédicteurs ###
wdbc <- read.table("wdbc.data.txt", sep = ",")

n <- dim(wdbc)[1]
p <- dim(wdbc)[2]
#X <- wdbc[, c(1, 3:31)]
X <- wdbc[, 3:p]
Y <- wdbc[, 2]
Kmax <- 10
erroremp <- rep(0, Kmax)
for (k in 1:Kmax) {
  Ypred = knn(train = X[1:n, ], test = X[1:n, ], cl = Y[1:n], k = k)
  erroremp[k] = mean(Y[1:n] != Ypred)
}
plot(erroremp, type = "l")

# Validation croisée 
# n = 569 = 114 ∗ 5 − 1
Itrain <- function(i) return((1 + (i-1) * 114):(min(n, i * 114))) # Base d'entrainement
Itest <- function(i) {
  output = c()
  for (j in 1:5) if (j!=i) output = c(output,Itrain(j))
  return(output) 
}

error <- rep(0, Kmax)

for (i in 1:5) { # 5 boucles pour 5 échantillons
  train <- Itrain(i)
  test <- Itest(i)
  for (k in 1:Kmax) {
    Ypred = knn(train=X[train,],test=X[test,],cl=Y[train],k=k) 
    error[k] = error[k] + mean(Ypred!=Y[test])/5 # On fait 5 validations
  }
}

plot(error, type="l")
 
### Arbres de classification et forêts aléatoires ###
library(tree)
a = read.table("ronflement.txt", header = TRUE)
a$RONFLE = as.factor(a$RONFLE)

# Création d'un arbre
r <- tree(RONFLE~AGE + ALCOOL + TABA, data = a)
#r
#summary(r)
plot(r) # Afficher l'arbre
text(r) # Afficher les règles de décision

# Prédictions
Xnew = data.frame(AGE=35,ALCOOL=1.5,TABA=1)
#predict(r, Xnew, type = "class") # Affiche uniquement la classe prédite
#predict(r, Xnew)

# Imposer le nombre de feuilles
r2 <- prune.tree(r, best = 4)
plot(r2) ; text(r2)

# Validation croisée pour le nombre de feuilles
r3 = cv.tree(r)
plot(r3$size, r3$dev, type = "b") # La meilleure taille semble être 7

r <- tree(RONFLE~AGE+ALCOOL+TABA,data=a)
r2 <- prune.tree(r,best=7) # Imposer la taille optimale

# Random Forest
library(randomForest)
r <- randomForest(RONFLE~AGE+ALCOOL+TABA,data=a) 
#r
predict(r, Xnew)

# Nombre d'arbres dans la forêt : par défaut : B = 500
r <- randomForest(RONFLE~AGE+ALCOOL+TABA, data = a, ntree = 100)
plot(r) # Affiche un graphique de l'erreur

##### Régression ###
### Moindres carrés ###
library(gdata)
concrete <- read.xls("Concrete_Data.xlsx", header = TRUE)
X <- concrete[, 1:8]
Y <- concrete[, 9]
r <- lm(Y~., data = X)
#r
#summary(r) # Voir les détails de la régression
r <- lm(Y~.-1,data=X) # Y~. -1 permet d'enlever l'ordonnée à l'origine

# Prédictions sous la forme
# predict(r,newdata=Xnew)

### Estimateur ridge et LASSO ###
wine <- read.csv2("/Users/okabeshu/Documents/ENSAE/Introduction au Machine Learning/Wine/winequality-red.csv")

library(lars)
X <- wine[, 1:11]
Y <- wine[, 12]
r <- lars(x = data.matrix(X), y = Y) # Ne marche pas avec as.matrix
plot(r) # Graphique des LASSO

library(glmnet)
r <- glmnet(x = as.matrix(X), y = Y, family = "gaussian")
#r$a0[11] # Intercept
#r$beta[, 11] # Coefficient
# Validation croisée
rcv <- cv.glmnet(x = data.matrix(X), y = Y, family = "gaussian", nfold = 5) # (cv.)glmnet prefers data.matrix to as.matrix
# nfold pour la validation croisée
plot(rcv)

# Prédictions avec glmnet
#rcv$lambda.min
r <- glmnet(x=as.matrix(X), y = Y, family = "gaussian", lambda = c(rcv$lambda.min))
#r$beta # Coefficients
#predict(r, newx=as.matrix(Xnew)) # Insérer un Xnew qui fonctionne


### LASSO logistique ###
n <- dim(wdbc)[1]
p <- dim(wdbc)[2]
Y <- wdbc[, 2]
X <- wdbc[, 3:p]
r <- glmnet(x = as.matrix(X), y = Y, family = "binomial") # LASSO logistique
rcv <- cv.glmnet(x = as.matrix(X), y = Y, family = "binomial", nfold = 5)
#rcv$lambda.min
plot(rcv)
r <- glmnet(x = as.matrix(X), y = Y, family="binomial", lambda = c(rcv$lambda.min))

# Prédiction sous la forme
#predict(r,newx=as.matrix(Xnew))
# Attention : R prédit X*beta : le label à prédire est le signe de cette quantité !

##### Apprentissage non supervisé #####
### ACP ###
#USArrests # BDD déjà installée
acp <- prcomp(USArrests)
#acp
# Attention : standard deviation : RACINE CARRÉE de lambda_j
plot(acp) # Histogramme des racines carrées de lambda
biplot(acp) # Résultat (projection) de l'ACP

# Normalisation des données
acp <- prcomp(USArrests, scale = TRUE)
plot(acp)
biplot(acp)

### k-means ###
# Iris
X <- iris[, 1:4]
Y <- iris[, 5]
res <- kmeans(X, centers = 3) # centers doit être un nombre ou un vecteur 
#res$centers
res$cluster # Résultats du clustering
table(res$cluster, Y)


# Movie Lens 100K (il existe plusieurs versions de Movie Lens : prendre le 100K (vieux))
library(softImpute) # Matrix Completion via Iterative Soft-Thresholded SVD : Complète les matrices
X <- read.table("/Users/okabeshu/Documents/ENSAE/Introduction au Machine Learning/Movie Lens 100K/u.data")#sep = ":", header = F, colClasses = c(NA, "NULL"))
# Dans cette BDD, les variables correspondent à : user id | item id | rating | timestamp

A <- Incomplete(i=X$V1,j=X$V2,x=X$V3) # i row indices, j column indices, x a vector of values
# Tous les utilisateurs n'ont pas noté tous les films, pour prédire on passe donc par softImpute
B <- softImpute(A,rank.max=5) # Remplit la matrice de type Incomplete avec des prédictions des valeurs manquantes
impute(B,3,1)
impute(B,4,1)

# Plusieurs prédictions d'un coup
i <- c(3,3,3,4,4,4)
j <- c(1,2,3,1,2,3)
impute(B,i,j)

# Compléter la matrice initiale
# Attention cela peut faire exploser la mémoire
Y <- complete(A,B)
Y_round <- round(Y)
A

# MSE en fonction du rang
data = X[1:80000,]
A = Incomplete(i=data$V1,j=data$V2,x=data$V3) # Créer la matrice de tous les utilisateurs et tous les films
test = X[80001:100000,]
MSE = c()
for (k in 1:10) {
  B <- softImpute(A, rank.max=k) # Tester pour différentes valeurs de k
  pred <- impute(object=B,i=test$V1,j=test$V2) # Prédictions
  MSE <- c(MSE, mean((pred-test$V3)^2)) # Mean Squared Error
}

plot(MSE, type = "l")
