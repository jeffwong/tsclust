\name{bottomk}
\alias{bottomk}
\title{
Get Bottom K Elements
}
\description{
Returns the indices of the smallest k elements from a vector
}
\usage{
bottomk(x, k)
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
topk
}
\examples{
x = rnorm(5)
indices = bottomk(x, k=2)
x[indices]
}

