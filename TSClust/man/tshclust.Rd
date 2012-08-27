\name{tshclust}
\alias{tshclust}
\title{
Hierarchical Clustering for Time Series
}
\description{
Hierarchical Clustering for Time Series
}
\usage{
tshclust(x, k, distmat="DTW", ...)
}
\arguments{
  \item{x}{
A matrix where each column represents a time series
}
  \item{k}{
An optional argument: specifies the number of clusters to use when cutting the 
tree
}
  \item{distmat}{
Character vector.  Determines how tshclust should compute the pairwise distance
matrix.  If "DTW", compute the distance matrix using DTW and the 'proxy' package
(found in R package 'dtw').  If "tsdist", use tsclust's distance matrix function
}
  \item{...}{
Additional arguments if distmat was set to "tsdist"
}
}
\details{
Uses custom distance functions to compute the pairwise distance matrix,
then runs hclust from the 'stats' package.  Possible to use tsdist with
R, C, CUDA, or the dist function from the 'proxy' package.  
}
\value{
The result of hclust when using tsdist as the dist object.  If k is specified, 
also returns the cluster labelings of x.
\item{hclust}{The result of hclust}
\item{clusters}{The cluster assignments}
}
\references{
}
\author{
Jeffrey Wong
}
\note{
}

\seealso{
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
tshclust(x)
}
\keyword{ hclust }
