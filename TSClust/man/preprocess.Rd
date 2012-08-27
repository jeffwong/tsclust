\name{preprocess}
\alias{preprocess}
\title{
Preprocess Time Series Data
}
\description{
Demean and standardize columns of a matrix
}
\usage{
preprocess(x, demean = TRUE, standardize = TRUE)
}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{x}{
Data where each column represents a time series
}
  \item{demean}{
True if the mean of each time series should be removed
}
  \item{standardize}{
True if the variance of each time series should be standardized
}
}
\details{
}
\value{
The transformed matrix
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

x = preprocess(x, demean=TRUE, standardize=FALSE) 
}

