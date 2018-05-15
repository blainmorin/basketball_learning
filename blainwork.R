
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
outcomes = NBATrainSLBD %>%
  select(Won.Home, HFinal, VFinal)
outcomes = as.matrix(outcomes)

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





### Run model with parallelization

cl = makeCluster(detectCores())
registerDoParallel(cl)

predictions = foreach(i = 1:ncol(outcomes), .combine = cbind, .packages = "caret") %dopar% {
  
  lasso = train(x = x, y = outcomes[ , i],
                method = "glmnet", tuneGrid = parameter.values)
  lasso.predictions = predict.train(lasso, newdata = NBATestSLBD)
  
  
}

stopCluster(cl)
