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
#' Print quickSentiment Evaluation Results
#'
#' @param x An object of class `quickSentiment_eval`.
#' @param ... Further arguments passed to or from other methods.
#' @export
print.quickSentiment_eval <- function(x, ...) {

  cat("=========================================\n")
  cat(" quickSentiment Model Evaluation \n")
  cat("=========================================\n")
  cat("Target Class:  ", x$target_class, "\n\n")

  cat("--- Global Metrics ---\n")
  cat("ROC AUC:       ", round(x$auc_roc, 4), "\n")
  cat("PR AUC:        ", round(x$auc_pr, 4), "\n\n")

  cat("--- Optimal Thresholds ---\n")
  cat("Best ROC Threshold (Youden's J): ", round(x$best_threshold_roc, 4), "\n")
  cat("Best PR Threshold (F1-Score):    ", round(x$best_threshold_pr, 4), "\n")
  cat("Accuracy at Best PR Threshold:   ", round(x$accuracy_at_best, 4), "\n\n")

  cat("--- Threshold Summary Table ---\n")
  # Print the user's custom table, rounded nicely to 3 decimal places
  print(round(x$threshold_summary, 3), row.names = FALSE)

  cat("\n(Note: Use plot() to view ROC and PR curves)\n")

  # Return the object invisibly so the data isn't lost
  invisible(x)
}
