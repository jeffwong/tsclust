setClass("tsSegments", representation = "list", S3methods=T)

mts2tsSegments = function(mts) {
  structure(list(mts = mts, ...), class="tsSegments")
}
