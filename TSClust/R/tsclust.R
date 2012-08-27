#Last updated July 13 2011.  See tsknn function
#for problems that still need to be solved

#Each column of x should represent a subject
#The rows represent observations in time
preprocess = function(x, demean=TRUE, standardize=TRUE) {
  apply(x, 2, function(j, demean, standardize) {
    if(demean) {
      j = j - mean(j)
    }
    if(standardize) {
      j = j / sd(j)
    }
    return (j)
  }, demean=demean, standardize=standardize)
}

topk = function(x,k) {
  x.order = order(x, decreasing=T)
  x.order[1:k]
}

bottomk = function(x,k) {
  x.order = order(x)
  x.order[1:k]
}

.dist.euclidean = function(x,y) {
  sqrt(sum((x-y)^2))
}

.dist.dtw = function(x,y) {
  dtw(x,y)$normalizedDistance
  #dtw(x,y)$distance
}

.dist.jure = function(x,y) {
  beta.hat = cor(x,y) * (sd(y) / sd(x))
  alpha.hat = mean(y) - (beta.hat * mean(x))
  y.fitted = alpha.hat + beta.hat * x
  sqrt(sum((y - y.fitted)^2))
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

#Public function to run kmeans on time series data
tskmeans = function(x, distfn=.dist.dtw, k=2, num.iters = 10, method="R", verbose=F) {
  if(method=="R")
  {
    dimension = nrow(x)
    centers = sapply(1:k, function(i, dimension) {
      rnorm(dimension,0,1)
    }, dimension=dimension)
    centers.last = centers
    assignments = rep(0, ncol(x))
  
    for(z in 1:num.iters) {
      if(verbose)
      {
        print(paste("Begin iteration", z, "of", num.iters, sep=" "))
      }
      assignments = apply(x, 2, function(column, centers) {
        distanceToCenters = apply(centers, 2, function(center, column) {
          distfn(center, column)
        }, column=column)
        index = which.min(distanceToCenters)
        return (index)
      }, centers=centers) 
      centers = sapply(1:k, function(i, x, centers.last, assignments) {
        indices = which(assignments == i)
        if(length(indices) > 1) {
          cluster = x[,indices]
          apply(cluster, 1, mean)
        }
        else {
          centers.last[,i]
        }
      }, x, centers.last=centers.last, assignments=assignments)
      if(sum((centers - centers.last)^2) < 1) {
        break
      }
      centers.last = centers
    }

    withinss = sapply(1:k, function(i, x, centers, assignments) {
      indices = which(assignments == i)
      cluster = x[,indices]  #columns that belong to ith cluster
      cluster.d = apply(cluster,2,function(column, center) {
        distfn(column, center)
      }, center=centers[,i])
      sum(cluster.d^2)
    }, x=x, centers=centers, assignments=assignments)
    
    return (list(centers=centers, clusters=assignments, withinss=withinss))
  }
  else if(method == "C")
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

    x.kmeans = .C('RKMeans', x.data, centers=centers, numTS, tslength, k, 
      clusters=clusters, withinss = withinss, success=success)
    list(centers = x.kmeans$centers, clusters = x.kmeans$clusters,
      withinss = x.kmeans$withinss, success = x.kmeans$success)
  }
  else if(method == "CUDA")
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

#Public function to run hclust on time series data
#Note that the hclust algorithm expects the data to be normalized
tshclust = function(x, k, distmat="DTW", ...) {
  x = preprocess(x, demean=T, standardize=F)
  if(distmat == "DTW") {
    x.dist = dist(x, method="DTW", by_rows=F)
  } else if(distmat == "tsdist") {
    x.dist = tsdist(x, ...)
  }
  x.hclust = hclust(x.dist)
  plot(x.hclust)

  if(!missing(k)) {
    x.hclust.labels = cutree(x.hclust, k=k)
    return (list(hclust = x.hclust, clusters=x.hclust.labels))
  }
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
