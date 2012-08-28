#Last updated July 13 2011.  See tsknn function
#for problems that still need to be solved

#Public function to run kmeans on time series data
tskmeans = function(x, k=2, num.iters = 10, gpu=F, verbose=F) {
  if(!gpu)
  {
    return (kmeans(t(x), k))
  }
  else
  {
    dimension = nrow(x)
    centers = sapply(1:k, function(i, dimension) {
      rnorm(dimension,0,1)
    }, dimension=dimension)

    x.data = as.single(as.vector(x))
    centers = as.single(as.vector(centers))
    numTS = as.integer(ncol(x))
    tslength = as.integer(nrow(x))
    k = as.integer(k)

    clusters = as.single(rep(0, ncol(x)))
    withinss = as.single(rep(0, k))
    success = integer(1)  
 
    x.kmeans = .C('cudaRKMeans', x.data, centers=centers, numTS, tslength, k, 
      clusters=clusters, withinss = withinss, success=success)
    list(centers = x.kmeans$centers, clusters = x.kmeans$clusters,
      withinss = x.kmeans$withinss, success = x.kmeans$success)
  }
}


#Public function to calculate the distance between
#multiple time series
tsdist = function(x, distfn=.dist.dtw, method="R") {
  distancematrix = matrix(0, ncol(x), ncol(x))
  if(method == "R")
  {
    combinations = combn(1:ncol(x), 2)
    distances = apply(combinations, 2, function(j) {
      col1 = j[1]
      col2 = j[2]
      distfn(x[,col1], x[,col2])
    })
    distancematrix[col(distancematrix) < row(distancematrix)] = (distances)
    as.dist(distancematrix)
  }
  else if(method == "C")
  {
    x.c = as.single(as.vector(x))
    x.distances = single( ncol(x)/2 * (ncol(x) - 1) )
    numTS = as.integer(ncol(x))
    tslength = as.integer(nrow(x))
    result = .C('RTSDist', x.c, numTS, tslength, distances=x.distances)
    distancematrix[col(distancematrix) > row(distancematrix)] = result$distances
    as.dist(t(distancematrix)) 
  }
  else if(method == "CUDA")
  {
    x.c = as.single(as.vector(x))
    x.distances = single( ncol(x)/2 * (ncol(x) - 1) )
    numTS = as.integer(ncol(x))
    tslength = as.integer(nrow(x))
    success = integer(1)
    result = .C('cudaRTSDist', x.c, numTS, tslength, distances=x.distances,
      success=success)
    distancematrix[col(distancematrix) > row(distancematrix)] = result$distances
    as.dist(t(distancematrix)) 
  }
}

#Public function to run hclust on time series data
#Note that the hclust algorithm expects the data to be normalized
tshclust = function(x, k, distfn, ...) {
  x = preprocess(x, demean=T, standardize=F)
  x.dist = tsdist(x, ...)
  x.hclust = hclust(x.dist)
  return (x.hclust)
}

#Public function to compute knn using time series data
#data and label belong to the training set
#y is a multivariate time series matrix that we want to label

#When referring to the labels of the k-neighbors, there is frequently
#a tie for the mode.  Need an intelligent way to break tie
#e.g. if k=4, the 4 neighbors may be labeled A A B B
tsknn = function(data, label, k, y, distfn=.dist.dtw) {
  apply(rbind(1:ncol(y), y), 2, function(j) {
    colIndex = j[1]
    distances = apply(data, 2, function(x,y) {  #computes distances between y and data
      distfn(x,y)
    },y=j[-1])
    smallestDistance.indices = bottomk(distances,k)
    label.table = table(label[smallestDistance.indices])
    names(label.table[which.max(label.table)])
  })
}
