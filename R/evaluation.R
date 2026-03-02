#' Calculate Classification Metrics
#'
#' A lightweight, dependency-free alternative to caret::confusionMatrix.
#' Calculates accuracy, and for binary classification, adds precision, 
#' recall, and F1 score.
#'
#' @param predicted A factor vector of predicted classes.
#' @param actual A factor vector of actual true classes.
#'
#' @return A list object containing the following metrics:
#'   \item{confusion_matrix}{A base R `table` object representing the cross-tabulation of predictions vs. actuals.}
#'   \item{accuracy}{Numeric. The overall accuracy of the model.}
#'   \item{precision}{Numeric. (Binary only) The Positive Predictive Value.}
#'   \item{recall}{Numeric. (Binary only) The True Positive Rate (Sensitivity).}
#'   \item{specificity}{Numeric. (Binary only) The True Negative Rate.}
#'   \item{f1_score}{Numeric. (Binary only) The harmonic mean of precision and recall.}
#'
#' @export
#' 
evaluate_metrics <- function(predicted, actual) {
  
  # Ensure factors have the same levels to prevent table dimension mismatches
  levels_union <- union(levels(predicted), levels(actual))
  predicted <- factor(predicted, levels = levels_union)
  actual <- factor(actual, levels = levels_union)
  
  # 1. Base R Confusion Matrix
  cm <- table(Predicted = predicted, Actual = actual)
  
  # 2. Overall Accuracy
  accuracy <- sum(diag(cm)) / sum(cm)
  
  results <- list(
    confusion_matrix = cm,
    accuracy = accuracy
  )
  
  # 3. Binary Specific Metrics (Precision, Recall, F1)
  if (length(levels_union) == 2) {
    # Assuming the second level is the "Positive" class (Standard behavior)
    tp <- cm[2, 2]
    fp <- cm[2, 1]
    fn <- cm[1, 2]
    tn <- cm[1, 1]
    
    precision <- ifelse((tp + fp) == 0, 0, tp / (tp + fp))
    recall <- ifelse((tp + fn) == 0, 0, tp / (tp + fn)) # Also called Sensitivity
    specificity <- ifelse((tn + fp) == 0, 0, tn / (tn + fp))
    f1_score <- ifelse((precision + recall) == 0, 0, 2 * (precision * recall) / (precision + recall))
    
    results$precision <- precision
    results$recall <- recall
    results$specificity <- specificity
    results$f1_score <- f1_score
  }
  
  return(results)
}