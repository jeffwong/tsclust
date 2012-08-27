\name{TSClust-package}
\alias{TSClust-package}
\alias{TSClust}
\docType{package}
\title{
Time Series Clustering and Classification
}
\description{
Clustering and classification for time series data using
Dynamic Time Warping
}
\details{
\tabular{ll}{
Package: \tab TSClust\cr
Type: \tab Package\cr
Version: \tab 1.0\cr
Date: \tab 2011-05-27\cr
License: \tab GPL\cr
LazyLoad: \tab yes\cr
}

Computing the Dynamic Time Warp metric can take a long time.  Since
this metric frequently needs to be computed between (n choose 2) pairs of 
time series, we offer a parallelized version that runs in CUDA.  
To take full advantage of this package, be sure to have the latest version of 
CUDA installed.

Very few functions:
tsdist - compute the distance matrix corresponding to a multivariate 
time series matrix
tskmeans - runs a parallel implementation of kmeans using the DTW metric
tsknn - utilizes tsdist to compute pairwise distances, then runs kNN
tshclust - utilizes tsdist to compute pairwise distances, then runs hclust

}
\author{
Jeffrey Wong

Maintainer: <jeff.ct.wong@stanford.edu>

}
\references{

}

\keyword{ package }
\keyword{ timeseries }

\seealso{

}
\examples{

}
