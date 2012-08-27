\name{tsdist}
\alias{tsdist}
\title{
Time Series Distance Matrix
}
\description{
Computes pairwise distances of time series objects
}
\usage{
tsdist(x, distfn=.dist.ts, method="R")
}
\arguments{
  \item{x}{
A matrix where each column is a time series
}
  \item{distfn}{
A function that returns the distance between two time series
}
  \item{method}{
A character vector.  If "R", uses an R implementation to calculate the distance 
matrix using the supplied distance function.
If "C", uses a C implementation to calculate the distance matrix using DTW.
If "CUDA", uses a parallel CUDA implementation to calculate the distance matrix
using DTW.
}
}
\details{
Computes all (n choose 2) distances and formats the result as an R distance
object.  distfn should be a function of the form function(x,y), and should
return a single numeric.  TSClust contains two predefined distance functions,
dist.ts which uses DTW, and dist.euclidean, which is the euclidean distance
of two vectors.
}
\value{
A distance matrix, akin to calling 'dist(x)', but uses dynamic time warp
}
\references{
}
\author{
Jeffrey Wong
}
\note{
}

\seealso{
dist
}
\examples{
numA = 25
numB = 25

ts.sim1 = sapply(1:numA, function(i) {              
  arima.sim(n = 50, list(ar = c(0.8, -0.5), ma=c(-0.23, 0.25)) )
})
ts.sim2 = sapply(1:numB, function(i) {              
  20 * arima.sim(n = 50, list(ar = 0.2))
})
x = matrix(c(ts.sim1, ts.sim2), 50, numA+numB) 
tsdist(x)
}

\keyword{ dist }
