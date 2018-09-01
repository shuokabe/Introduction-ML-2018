rm(list = ls())

setwd("/Users/okabeshu/Documents/ENSAE/Introduction au Machine Learning/Données TD") # Path of the main folder


### TD 1 ###
# Question 1
# Données USPS
train = read.table("train.txt", header=F)
test = read.table("test.txt", header=F)

trainY = as.matrix(train[,1])
trainX = as.matrix(train[,2:257])
testY = as.matrix(test[,1])
testX = as.matrix(test[,2:257])

# Question 2
# En entrée un vecteur de taille 256
f = function(v) image(matrix(data=v,nc=16,nr=16)[1:16,16:1],col=gray((100:0)/100))

# Question 3

trois = colMeans(trainX[trainY==3,]) # Moyenne sur les valeurs des colonnes 
f(trois)

# Question 4 (attention, ça peut prendre quelques minutes) : sur la base d'apprentissage
library(class)

perftrain = rep(0,10)
for (K in 1:10) {
  trainYpred = knn(train=trainX,test=trainX,cl=trainY,k=K)
  perftrain[K] = mean(trainYpred!=trainY) # Erreur de prédictions
}
plot(perftrain,type="l")

# Question 5 : même question pour la base de test

perftest = rep(0,10)
for (K in 1:10) {
  testYpred = knn(train=trainX,test=testX,cl=trainY,k=K)
  perftest[K] = mean(testYpred!=testY)
}
plot(perftest,type="l")

# Question 6

plot(perftest, type="l")
lines(perftrain, col="red", type="l")

### TD 2 ###
# EXERCICE 2
# Question 1

X = iris[,1:4]
Y = as.character(iris[, 5])
Y[Y!="versicolor"] = "autre" # Fusionner les autres labels
Y = as.factor(Y)

# Question 2

perm = sample(1:150) # Permutation aléatoire
X = X[perm, ]
Y = Y[perm]

# Découpage en base d'apprentissage et en base de test de taille égale
Xtrain = X[1:75,]
Xtest = X[76:150,]
Ytrain = Y[1:75]
Ytest = Y[76:150]

# Question 3

library(tree)
arbre = tree(formula = Ytrain ~ ., data = Xtrain) # Arbre de classification
summary(arbre)    # dernière ligne pour la proportion d'erreur demandée

# Question 4
# Représentation graphique
plot(arbre)
text(arbre)

# Question 5
# Prédiction sur le label de test
Ypred = predict(arbre, newdata=Xtest, type="class")
mean(Ypred!=Ytest) # Erreur commise sur l’échantillon de test

# EXERCICE 3
# Question 1

Train = read.csv2("DataTrain.csv", header=TRUE)
Test = read.csv2("DataTest.csv", header=TRUE)
Train$Y = as.factor(Train$Y)
Test$Y = as.factor(Test$Y)

# Question 2

library(tree)
arbre <- tree(Train$Y~X1+X2+X3, data=Train[,2:4]) # Ne fonctionne pas
Ypred = predict(arbre, newdata=Test[,2:4],type="class")
mean(Ypred!=Test[,1])

# Question 3

library(class)
Ypred2 = knn(train=Train[,2:4],test=Test[,2:4],cl=Train[,1],k=3)
mean(Ypred2!=Test[,1])

# Question 4

pairs(Train[,2:4], col=Train[,1])

# Question 5

library(randomForest)
foret = randomForest(Y~X1+X2+X3,data=Train) # Ne marche pas non plus
Ypred3 = predict(foret,newdata=Test[,2:4])
mean(Ypred3!=Test[,1])

# Question 6

# On a comparé 3 prédicteurs dont les risques sur l'échantillon de test sont:
# 0.4479 (arbre)
# 0.1472 (3-nn)
# 0.1288 (random forest)

# La borne de Hoeffding vue en cours pour majorer la probabilité d’erreur hors- échantillon 
# de la méthode que vous aurez sélectionnée avec une probabilité 90%.

borne = sqrt(2*log(2*3/0.1)/length(Test$Y))
0.1288 + borne

### TD 3 ###
# EXERCICE 2

# Question 1

A = read.table("trees.txt",header=TRUE)
X = A[,1:2]
Y = A[,3]

# Question 2

# Ximplement: volume = cste.hauteur.rayon^2
# donc log(volume) = log(cste) + log(hauteur) + 2.log(rayon)
# donc on passe les données au log

XL = log(X)
YL = log(Y)

# Question 3

library(lars)
n = dim(X)[1]
XLbar = cbind(diag(n), as.matrix(XL)) # diag(n) : matrice identité
lasso = lars(x = XLbar, y = YL)
plot(lasso)

# On voit que la 18ème observation a un coefficient bien plus grand que les autres
# Ceci est par exemple confirmé par un test de Student dans le modèle linéaire

summary(lm(YL~XLbar[,32]+XLbar[,33]+XLbar[,18]))

# Question 4

XL = XL[c(1:17,19:31), ]
YL = YL[-18]
summary(lm(YL~XL[,1]+XL[,2]))

# En particulier, voir le R^2 = 0.98, si on compare avec le modèle linéaire sans passage au log

X = X[c(1:17,19:31), ]
Y = Y[c(1:17,19:31)]
summary(lm(Y~X[,1]+X[,2]))

# on voit que le R^2 ne vaut que 0.95, ce qui montre qu'on a sans doute eu raison...

# EXERCICE 3

# question 1

Lasso <- function(X,Y, lambda,critereconv=0.0001, intercept=TRUE) {
  n = dim(X)[1]
  p = dim(X)[2]
  beta = rep(0, p)
  gap = Inf
  a = 0
  while (gap > critereconv) {
    betaprev = beta
    if (intercept) a = mean(Y- X %*% beta) # %*% : produit matriciel
    for (j in 1:p) {
      betaJ = beta
      betaJ[j] = 0
      beta[j] = sum(Y- X %*% betaJ - a) * X[, j]#)
      if (abs(beta[j])>lambda/2) {
        beta[j] = sign(beta[j])*(abs(beta[j])-lambda/2)/sum(X[,j]^2)
      } else {
          beta[j] = 0
      }
    }
    gap = max(abs(betaprev - beta))
  }
  return(list(beta = beta, intercept = a))
}

# Question 2

# Par exemple jouer avec lambda pour voir que si lambda est assez grand on tue le deuxième paramètre
Lasso(X=as.matrix(X),Y=Y,lambda=0.1)
# d'un autre côté, si lambda est nul, on retrouve l'estimateur des moindres carrés
Lasso(X=as.matrix(X),Y=Y,lambda=0)
lm(Y~as.matrix(X))


### TD 4 ###
# EXERCICE 1
set.seed(42)
# Question 1
# Exemple jouet
n = 20
p = 8
beta = c(5,3,0,0,1.5,0,0,0)
X = matrix(data = rnorm(p * n), nr = n, nc = p)
Y = X %*% beta + rnorm(n, 0, 1) # <- représente epsilon_i 

# Question 2
library(glmnet)
lasso <- glmnet(x = X, y = Y, intercept=FALSE, lambda=(1:100)/100) # On calcule l'estimateur Lasso pour lambda jusqu'à 1

# Question 3

err <- colSums((X %*% lasso$beta - X %*% beta)^2) # Fonction à tracer
plot(lasso$lambda, err)
which.min(err) # Donne l'indice du minimum #s88 89

# Question 4

cvlasso = cv.glmnet(x=X, y=Y, nfold=5) # Validation croisée
plot(cvlasso) # Représentation graphique de l'erreur quadratique
cvlasso

#The two different values of λ reflect two common choices for λ. The λmin is the one which minimizes out-of-sample loss in CV. The λ1se is the one which is the largest λ value within 1 standard error of λmin. One line of reasoning suggests using λ1se because it hedges against overfitting by selecting a larger λ value than the min. Which choice is best is context-dependent.
#Confidence intervals represent error estimates for the loss metric (red dots). They're computed using CV. The vertical lines show the locations of λmin and λ1se. The numbers across the top are the number of nonzero coefficient estimates.

# Question 5

beta0 = colSums(lasso$beta!=0) # Nb de composantes non nulles pour un lambda fixé
plot(lasso$lambda, beta0)

# Commentaire : il n'y a pas de raison pour que le paramètre lambda qui marche le mieux
# en prédiction soit aussi celui qui marche le mieux pour retrouver les bonnes variables!!

# EXERCICE 3
# Question 1
A = read.table("/Users/okabeshu/Documents/ENSAE/Introduction au Machine Learning/Tests fonction du cours/wdbc.data.txt",sep=",")
n = dim(A)[1]
p = dim(A)[2]

Y = A[,2]
X = A[,3:p]

# Question 2

library(glmnet)
lLasso = glmnet(x=as.matrix(X), y=Y,intercept=TRUE, family="binomial") # binomial pour le logistique
plot(lLasso)
# Commentaire: on ne voit pas grand chose, on peut "zoomer" sur les modèles à peu de coefficients :
plot(lLasso,xlim=c(1,50), ylim=c(-20,20))
# Pour savoir à quelles variables ça correspond:
lLasso$beta[]
# On voit que les premières variables incluses dans le modèle sont V30, V25 puis V23. 
# Cependant V25 disparaît ensuite. D'où la difficulté d'interpréter ce type de résultats...

# Question 3

cvlLasso = cv.glmnet(x=as.matrix(X),y=Y,intercept=TRUE, family="binomial", nfold=5) # Avec validation croisée
plot(cvlLasso)
# On voit que le minimum en log(Lambda) est quelque part entre -7 et -6...
cvlLasso
# lambda = 0.00119
optLasso = glmnet(x=as.matrix(X),y=Y,intercept=TRUE,family="binomial",lambda=c(0.00119))
optLasso$beta
# Les variables dans le modèle: 8,9,10,12,13,14,17,18,21,22,23,24,26,27,29,30,31

# Question 4

# On aimerait avoir une interprétation simple: variables sélectionnées = variables liées au type de cancer
# Malheureusement l'exercice 1 montre que ça n'est pas si simple... on a un modèle utile pour la prédiction, mais vraisemblablement trop de variables...
# En effet, estimons le modèle logistique avec les variables sélectionnées

Xlasso = as.matrix(X)[,c(8,9,10,12,13,14,17,18,21,22,23,24,26,27,29,30,31)-2] # car les variables sont en fait numérotées à partir de 3...
r = glm(Y~Xlasso,family=binomial)
summary(r)

# On voit qu'il y a... beaucoup de coefficients non significatifs. L'optimalité en prédiction ne signifie pas que l'on a trouvé les bonnes variables, ce sont deux problèmes différents!

### TD 5 ###
###Exercice 1###
# Question 1
red_wine = read.table("/Users/okabeshu/Documents/ENSAE/Introduction au Machine Learning/Wine/winequality-red.csv", sep=";", header = T)

# Question 2

red_acp = prcomp(red_wine, scale=T) # ACP avec normalisation des données
red_acp
plot(red_acp) # Affiche l'histogramme des racines carrées des lambda (variance)

variances = red_acp$sdev**2
cat("Variance expliquée par les deux premières directions : ", sum(variances[1:2]) / sum(variances))
biplot(red_acp, xlabs = rep("", nrow(red_wine))) # Résultat de l'ACP

# Positivement correlé au goût : alcohol, négativement : free.sulfur.dioxide, sulfur.dioxide

# Question 3
white_wine = read.table("/Users/okabeshu/Documents/ENSAE/Introduction au Machine Learning/Wine/winequality-white.csv", sep=";", header = T)

white_acp = prcomp(white_wine, scale=T)
white_acp
plot(white_acp)

variances = white_acp$sdev**2
cat("Variance expliquée par les deux premières directions : ", sum(variances[1:2]) / sum(variances))
biplot(white_acp, xlabs=rep("", nrow(white_wine)))
# Positivement correlé au goût : alcohol, négativement : volatile.acidity, chlorides (pas les mêmes)


### Exercice 2###

# J'ai mis avec le TP 5 deux bases de données, la base movielens '100k' et une plus grande '1M'.
# attention, de façon curieuse, movielens ne les fournit pas au même format: dans la première, l'ordres des observations est déjà  randomisé, pas dans l'autre...
# donc il faut randomiser soi-même les données avant de découper les échantillons de test et d'apprentissage
# de plus, dans la 1ère, le séparateur des observations est une tabulation, dans la seconde, c'est ":"

# on propose la solution à la question 1 et 2 pour la base '1M' car l'algo est assez rapide et que le cas '100k' a été traité en cours
# par contre pour les plus proches voisins, on passe à la base plus petite car l'algorithme est très lent

# Question 1

ratings = read.table("/Users/okabeshu/Documents/ENSAE/Introduction au Machine Learning/Movie Lens/ratings.dat", sep=":", header=F, colClasses = c(NA, "NULL"))
ratings = ratings[sample(1:nrow(ratings)),]

train = ratings[1:750000,]
test = ratings[750001:1000000,]

# Question 2
# Approche naïve : prédire la note par la moyenne par film
mean_ratings = aggregate(train[,3], list(train[,2]), mean) # Faire la moyenne de la note des films (train[,3]) pour chaque film
preds = mean_ratings[test[,2], 2]
cat("Mean Square Error : ", mean((preds-test[,3])^2, na.rm=T)) # Erreur quadratique moyenne

# Question 3
#install.packages("softImpute")
library(softImpute) # Méthode de factorisation

A = Incomplete(i=train$V1,j=train$V3, x=train$V5)
MSE = c()
for (k in 1:10){
  B = softImpute(A, rank.max=k, maxit = 500)
  pred = impute(object=B,i=test$V1,j=test$V3)
  pred = pmax(pmin(pred, 5), 1) # Pour être certain d'avoir des nombres entre 1 et 5 (tronquer puis arrondir)
  MSE = c(MSE, mean((pred-test[,3])^2)) # Erreur quadratique moyenne
}

MSE

# Remarque: avec la base 100k utilisée en cours, on ne pouvait "apprendre" correctement qu'une décomposition de rang 2 ou 3
# ici, on monte jusqu'au rang 8

# Question 4

# Warning!! le code est correct, mais ça prend des heures - pour un résultat qui est moins bon que celui obtenu avec la factorisation
# de matrices. Si vous voulez absolument le faire tourner, commencer déjé  par remplacer A par la base de données '100k' utilisée
# en cours en utilisant les instruction:
X = read.table("u.data", header=FALSE)
data = X[1:80000,]
A = Incomplete(i=data$V1,j=data$V2,x=data$V3)
test = X[80001:100000,]
# Avec la base 100K, lancer le programme, aller regarder un film et venir voir le résultat après. le temps d'exécution chez moi est d'environ
# 1h et quelques. Avec la base 1M, lancer le code, partir en week-end et regarder le résultat au retour - le temps d'execution est entre 50h
# et 100h.

# return top-k similar users
getKNN = function(A, B, i, k) {
  distance = rep(Inf,nrow(B))
  for(j in 1:nrow(B)) {
    idxs = intersect(which(A[i,] != 0), which(B[j,] != 0))
    if (length(idxs) > 0) distance[j] = sqrt(sum((A[i, idxs] - B[j, idxs])^2) / length(idxs))
  }
  if(min(distance) < Inf){
    idx = which.min(distance)
    return(idx)
  }
  else return(-1)
}

kNNRecommender = function(A, indices) {
  reco = rep(3, nrow(indices))
  for(i in 1:nrow(indices)){
    u = indices[i,1]
    m = indices[i,2]
    if (A[u,m] == 0){ #user u didn't see movie m
      ratings_m = A[which(A[,m]!=0),] #ratings of users who saw movie m, for all the movies they saw
      if (!is.null(nrow(ratings_m)) && nrow(ratings_m) > 0){
        nn = getKNN(A, ratings_m, u, 1)
        if (nn != -1){ #it could be the case that user u didn't see any movies that was seen by those who saw m
          reco[i] = ratings_m[nn, m] 
        }
      }
    }
  }
  return(reco)
}

preds = kNNRecommender(A, test[,1:2])
cat("Mean Square Error : ", mean((preds-test[,3])^2))

# Histoire de voir un résultat si vous n'avez pas des heures, faire tourner uniquement le programme 
# sur les 100 premières données de l'ensemble de test:

preds = kNNRecommender(A, test[1:100,1:2])
cat("Mean Square Error : ", mean((preds-test[,3])^2))

# Remarque : on peut changer la valeur par défaut quand on ne peut pas matcher



