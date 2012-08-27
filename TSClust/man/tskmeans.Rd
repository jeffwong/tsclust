\name{tskmeans}
\alias{tskmeans}
\title{
K-means for Time Series Data
}
\description{
K-means for Time Series Data
}
\usage{
tskmeans(x, distfn = .dist.ts, k = 2, num.iters = 10, method = "R", verbose=F)
}
\arguments{
  \item{x}{
A matrix where each column represents a time series
}
  \item{distfn}{
A function to calculate the distance between two time series
}
  \item{k}{
The number of clusters to initialize with
}
  \item{num.iters}{
An optional argument: sets the number of iterations for the kmeans algorithm.  
Default 10.
}
  \item{method}{
A character vector.  If "R", uses an R implementation.  If "C", uses a C 
implementation.  If "CUDA", uses a parallel CUDA implementation
}
  \item{verbose}{
A logical value.  If true, print information about the status of kmeans.
Only applicable when using method=R
}
}
\details{
Uses kmeans with the supplied distance function to compute cluster assignments.
Akin to the kmeans function from the stats package, but all computations are 
done natively in R, allowing the user to specify any distance function desired.  
If flexibility is not needed, specify method=C or method=CUDA to use external 
code to compute kmeans using the DTW distance function.  
}
\value{
\item{centers}{The centers of the clusters}
\item{clusters}{The cluster assignments for the columns of x}
\item{withinss}{The within sum of squares for each cluster}
\item{success}{1 if the C/CUDA call was successful, -1 if not}
}
\references{
}
\author{
Jeffrey Wong
}
\note{
}

\seealso{
kmeans
}
\examples{
require(dtw)

numA = 25
numB = 25

ts.sim1 = sapply(1:numA, function(i) {              
  arima.sim(n = 50, list(ar = c(0.8, -0.5), ma=c(-0.23, 0.25)) )
})
ts.sim2 = sapply(1:numB, function(i) {              
  20 * arima.sim(n = 50, list(ar = 0.2))
})
x = matrix(c(ts.sim1, ts.sim2), 50, numA+numB) 

x.kmeans = tskmeans(x, k=2)
x.clusters = x.kmeans$clusters
}
\keyword{ kmeans }
