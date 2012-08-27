require(TSClust)
require(e1071)

kmeans.table = function(clusters, clustersizes) {
  k = length(clustersizes)

  #all possible arrangements of labels
  label = permutations(k)  

  #all possible labels of data
  x.labels = apply(label, 1, function(x) {
    rep(x, times=clustersizes)
  })

  #Classification rates of all possible labels
  x.kmeans.rates = apply(x.labels, 2, function(x) {
    x.kmeans.table = as.matrix(table(x, clusters))
    x.kmeans.rate = sum(diag(x.kmeans.table)) / sum(clustersizes)
    x.kmeans.rate
  })

  index.max = which.max(x.kmeans.rates)

  table(x.labels[,index.max], clusters, dnn=c("true label", "predicted label"))
}

generateTSMatrix = function(clustersizes, tslength, p, q)
{
  k = length(clustersizes)
  tslist = sapply(1:k, function(i) {
    clustersize = clustersizes[i]
    ari = p[[i]]
    mai = q[[i]]
    ts.sim = sapply(1:clustersize, function(j) {
      arima.sim(n=tslength, list(ar=ari, ma=mai))
    })
    runif(1,1,20) * as.vector(ts.sim)
  })
  matrix(unlist(tslist), nrow=tslength, ncol=(length(unlist(tslist)) / tslength))
}

analyze = function(x, clustersizes) {
  par(ask=T)
  x.kmeans = tskmeans(x, k=length(clustersizes))
  x.kmeans.table = kmeans.table(x.kmeans$clusters, clustersizes)
  print(x.kmeans.table)

  x.kmeans = kmeans(t(x), centers=length(clustersizes))
  x.kmeans.table = kmeans.table(x.kmeans$cluster, clustersizes)
  print(x.kmeans.table)

  x.dist = tsdist(x)
  x.hclust = hclust(x.dist)
  plot(x.hclust)

  x.dist = dist(t(x))
  x.hclust = hclust(x.dist)
  plot(x.hclust)
}

#Euclidean k-means no good
set.seed(10)
x = generateTSMatrix(c(25,25),50, 
  list(c(0.8, -0.5), 0.2),
  list(c(-0.23, 0.25), 0))
analyze(x, c(25,25))

#Euclidean hclust no good
set.seed(50)
x = generateTSMatrix(c(20,30), 50,
  list(c(0.7,-0.2), 0.4),
  list(c(-.3, 0.1), -0.1))
analyze(x, c(20,30))

#3 Clusters, Euclidean k-means no good
set.seed(10)
x = generateTSMatrix(c(25,25, 10),50, 
  list(c(0.8, -0.5), 0.2, c(-0.1, 0.6)),
  list(c(-0.23, 0.25), 0, 0.5))
analyze(x, c(25,25,10))

