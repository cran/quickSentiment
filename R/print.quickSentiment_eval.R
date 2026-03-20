#' Plot ROC Curve
#'
#' @param x An object of class `quickSentiment_roc`.
#' @param ... Additional graphical parameters.
#' @importFrom graphics par plot points abline
#' @export
plot.quickSentiment_roc <- function(x, ...) {
  # Professional square margins
  oldpar <- par(no.readonly = TRUE)
  on.exit(par(oldpar))
  par(mar = c(4.5, 4.5, 3, 1.5), pty = "s")

  plot(x$curve$FPR, x$curve$TPR_Recall,
       type = "l", col = "darkred", lwd = 2,
       xlab = "False Positive Rate", ylab = "True Positive Rate",
       main = paste("ROC Curve (AUC =", round(x$auc, 3), ")"))

  abline(a = 0, b = 1, lty = 2, col = "gray")

  best_idx <- which(x$curve$Threshold == x$best_threshold)[1]
  points(x$curve$FPR[best_idx], x$curve$TPR_Recall[best_idx],
         col = "blue", pch = 19, cex = 1.5)
}


#' Plot Precision-Recall Curve
#'
#' @param x An object of class `quickSentiment_prc`.
#' @param ... Additional graphical parameters.
#' @importFrom graphics par plot points
#' @export
plot.quickSentiment_prc <- function(x, ...) {
  # Professional square margins
  oldpar <- par(no.readonly = TRUE)
  on.exit(par(oldpar))
  par(mar = c(4.5, 4.5, 3, 1.5), pty = "s")

  plot(x$curve$TPR_Recall, x$curve$Precision,
       type = "l", col = "darkblue", lwd = 2,
       xlab = "Recall (True Positive Rate)", ylab = "Precision",
       main = paste("PR Curve (AUC =", round(x$auc, 3), ")"))

  best_idx <- which(x$curve$Threshold == x$best_threshold)[1]
  points(x$curve$TPR_Recall[best_idx], x$curve$Precision[best_idx],
         col = "red", pch = 19, cex = 1.5)
}
