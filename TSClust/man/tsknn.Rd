\name{tsknn}
\alias{tsknn}
\title{
K-nearest neighbors classification for Time Series data
}
\description{
K-nearest neighbors classification for Time Series data
}
\usage{
tsknn(data, label, k, y, distfn = .dist.ts)
}
\arguments{
  \item{data}{
The training matrix where each column represents a time series
}
  \item{label}{
The labels for the columns of the training data
}
  \item{k}{
The number of neighbors to use in classification
}
  \item{y}{
A matrix of time series to classify.  Each column represents a time series
}
  \item{distfn}{
Distance function to use when comparing two time series
}
}
\details{
For every time series in y, compute the distance to all labeled points
in the training data.  The distance function is flexible, but is defaulted
to use DTW.  Finds the k nearest neighbors, and labels the time series
using the mode of the labels among the k neighbors.
}
\value{
The labels for the time series in y
}
\references{
}
\author{
Jeffrey Wong
}
\note{
Among the k nearest neighbors, there may not be a unique mode for the labels, 
i.e. if k = 4 the labels of the neighbors may be A A B B.  In this case, tsknn
does not have any intelligent way to break the tie for the mode of the labels.
}
\seealso{

}
\examples{
numA = 5;
numB = 5;
numC = 5;

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
label = as.factor( c(rep(0,numA), rep(1,numB), rep(2,numC)) )

ts.sim4 = sapply(1:5, function(i) {
  arima.sim(n=50, list(ar = 0.2, ma=0.4))
})
tsknn(x, label, 1, ts.sim4)

}
\keyword{ knn }
