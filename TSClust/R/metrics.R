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
