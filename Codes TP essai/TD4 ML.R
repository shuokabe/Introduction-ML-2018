rm(list = ls())
#install.packages("glmnet")
library(glmnet)
set.seed(1234)

### Exercice 1 ###
# 1
n = 20
p = 8
beta = c(5, 3, 0, 0, 1.5, 0, 0, 0)
X = matrix(rnorm(p*n), nr = n, nc = p)
Y = X %*% beta + rnorm(n) # Produit matriciel

# 2
help(glmnet)
lasso = glmnet(x = X, y = Y, intercept = F, lambda = (1:100)/100)

# 3
err = rep(NA, 100)
for (i in 1:100) {
  beta_hat = lasso$beta[ ,i]
  err[i] = sum((X%*%(beta_hat - beta))^2)
}
plot(lasso$lambda, err)

# 4
cv_lasso = cv.glmnet(x = X, y = Y, nfold = 5)
plot(cv_lasso)
cv_lasso$lambda.min

# 5
beta0 = colSums(lasso$beta != 0)
plot(lasso$lambda, beta0)

