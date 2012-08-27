\name{topk}
\alias{topk}
\title{
Get Top K Elements
}
\description{
Returns the indices of the largest k elements from a vector
}
\usage{
topk(x, k)
}
\arguments{
  \item{x}{
A vector
}
  \item{k}{
The amount of elements to return
}
}
\details{
}
\value{
The indices of the smallest k elements of x
}
\references{
}
\author{
Jeffrey Wong
}
\note{
}

\seealso{
bottomk
}
\examples{
x = rnorm(5)
indices = topk(x, k=2)
x[indices]
}
