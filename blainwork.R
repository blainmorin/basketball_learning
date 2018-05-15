
library(readr)
library(caret)
library(doParallel)
library(dplyr)


NBATrainSLBD <- read_csv("NBATrainSLBD.csv")
NBATestSLBD <- read_csv("NBATestSLBD.csv")


### Grid of parameter values to check
### alpha = 0 is ridge
### alpha = 1 is lasso
### alpha in between is elastic net
### lambda is our penalty 
parameter.values = expand.grid(alpha = seq(0,1, by = .1), lambda = 10^seq(3, -3, length.out = 300))




### Three outcomes to check
outcomes.wins = NBATrainSLBD %>%
  select(Won.Home)

outcomes.points = NBATrainSLBD %>%
  select(HFinal, VFinal)

outcomes.wins = as.matrix(outcomes.wins)
outcomes.points = as.matrix(outcomes.points)


### Specify Factors
factors = c("Home",
            "Visitor",
            "Season")

### Change columns to factors
NBATrainSLBD[factors] = as.data.frame(lapply(NBATrainSLBD[factors], factor))
NBATestSLBD[factors] = as.data.frame(lapply(NBATestSLBD[factors], factor))

### Input variables
x = NBATrainSLBD %>%
  select(-X1, -Date, -HFinal, -VFinal, -Won.Home, -GID, -CSpreadH)

### Change x to model matrix
x = model.matrix(~ ., data = x)
x = x[,-1] ### Removes intercept

### Change test data to model matrix
NBATestSLBD = model.matrix(~., data = NBATestSLBD)





### Run win model

cl = makeCluster(detectCores())
registerDoParallel(cl)

predictions.wins = foreach(i = 1:ncol(outcomes.wins), .combine = cbind, .packages = "caret") %dopar% {
  
  lasso = train(x = x, y = outcomes.wins[ , i],
                method = "glmnet", family = binomial, tuneGrid = parameter.values)
  lasso.predictions = predict.train(lasso, newdata = NBATestSLBD)
  
  
}

stopCluster(cl)

### Run points model

cl = makeCluster(detectCores())
registerDoParallel(cl)

predictions.points = foreach(i = 1:ncol(outcomes.points), .combine = cbind, .packages = "caret") %dopar% {
  
  lasso = train(x = x, y = outcomes.points[ , i],
                method = "glmnet", tuneGrid = parameter.values)
  lasso.predictions = predict.train(lasso, newdata = NBATestSLBD)
  
  
}

stopCluster(cl)



