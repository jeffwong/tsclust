require(TSClust)

set.seed(50)

numA = 10;
numB = 10;
numC = 10;

ts.sim1 = sapply(1:numA, function(i) {              
  arima.sim(n = 50, list(ar = c(0.8, -0.5), ma=c(-0.23, 0.25)) )
})
ts.sim2 = sapply(1:numB, function(i) {              
  arima.sim(n = 50, list(ar = 0.2, ma=0.4))
})
ts.sim3 = sapply(1:numC, function(i) {
  arima.sim(n=50, list(ma=0.8))
})

x = matrix(c(ts.sim1, ts.sim2, ts.sim3), 50, numA+numB+numC) 
label = as.factor( c(rep(1,numA), rep(2,numB), rep(3,numC)) )

#Generate series similar to 2
ts.sim4 = sapply(1:5, function(i) {
  arima.sim(n=50, list(ar = 0.2, ma=0.4))
})

#Generate series similar to 3
ts.sim5 = sapply(1:5, function(i) {
  arima.sim(n=50, list(ma=0.8))
})
y = cbind(ts.sim4, ts.sim5)

x = preprocess(x, T, T)
y = preprocess(y, T, T)

tsknn(x, label, k=3, y)

#Sometimes there is no unique mode for the labels of the knn
#Need more intelligent way to break ties
