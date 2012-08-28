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


