cl = makeCluster(detectCores())
registerDoParallel(cl)

wins.model.net = train(x = x, y = outcomes.wins$Won.Home,
                   method = "nnet", family = "binomial")

stopCluster(cl)

predictions.wins.net = predict.train(wins.model, newdata = NBATestSLBD, type = "prob")

logLoss(predictions.wins.net$`1`, NBATestSLBD$Won.Home)
