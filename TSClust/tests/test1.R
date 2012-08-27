require(TSClust)

numA = 25
numB = 25

ts.sim1 = sapply(1:numA, function(i) {              
  arima.sim(n = 50, list(ar = c(0.8, -0.5), ma=c(-0.23, 0.25)) )
})
ts.sim2 = sapply(1:numB, function(i) {              
  20 * arima.sim(n = 50, list(ar = 0.2))
})
x = matrix(c(ts.sim1, ts.sim2), 50, numA+numB) 
label = c(rep(1, numA), rep(2, numB))

#SHOW RESULTS WITHOUT PREPROCESSING

x.kmeans = tskmeans(x, k=2)
x.clusters = x.kmeans$clusters
table(label, x.clusters)

#RESULTS WITH PREPROCESSING

x = preprocess(x, standardize=T)
x.kmeans = tskmeans(x, k=2)
x.clusters = x.kmeans$clusters
table(label, x.clusters)
