#' Evaluate Model Performance (ROC and Precision-Recall)
#' @param predicted_probs Numeric vector of predicted probabilities for the positive class.
#' @param actual_classes Factor or character vector of the actual true labels.
#' @param positive_label Character string. The target class you want to evaluate.
#'
#' @return An object of class `quickSentiment_eval`, which is a list containing the following metrics:
#'   \item{target_class}{Character. The specific positive label used for the evaluation.}
#'   \item{auc_roc}{Numeric. The Area Under the Receiver Operating Characteristic curve.}
#'   \item{best_threshold_roc}{Numeric. The optimal probability threshold that maximizes Youden's J statistic.}
#'   \item{auc_pr}{Numeric. The Area Under the Precision-Recall curve.}
#'   \item{best_threshold_pr}{Numeric. The probability threshold that maximizes the F1-Score.}
#'   \item{accuracy_at_best}{Numeric. The overall accuracy of the model if `best_threshold_pr` is applied.}
#'   \item{roc}{S3 object containing the ROC curve data and metrics.}
#'   \item{prc}{S3 object containing the Precision-Recall curve data and metrics.}
#'   \item{threshold_summary}{A data frame summarizing Accuracy, Precision, Recall, and F1 at 0.1 threshold increments.}
#' @export
evaluate_performance <- function(predicted_probs, actual_classes, positive_label) {

  # 1. Vectorized Sorting
  ord <- order(predicted_probs, decreasing = TRUE)
  sorted_probs <- predicted_probs[ord]
  is_positive <- (actual_classes[ord] == positive_label)

  # 2. Cumulative sums
  tp_cum <- cumsum(is_positive)
  fp_cum <- cumsum(!is_positive)

  total_pos <- sum(is_positive)
  total_neg <- sum(!is_positive)

  # --- ROC Math ---
  tpr <- tp_cum / total_pos
  fpr <- fp_cum / total_neg
  specificity <- 1 - fpr
  youden_j <- tpr + specificity - 1

  dp_roc <- diff(fpr)
  mid_tpr <- (tpr[-1] + tpr[-length(tpr)]) / 2
  auc_roc <- sum(dp_roc * mid_tpr)

  # --- Precision-Recall Math ---
  precision <- tp_cum / (tp_cum + fp_cum)
  precision[is.nan(precision)] <- 0

  f1_score <- 2 * (precision * tpr) / (precision + tpr)
  f1_score[is.nan(f1_score)] <- 0

  dp_pr <- diff(tpr)
  mid_prec <- (precision[-1] + precision[-length(precision)]) / 2
  auc_pr <- sum(dp_pr * mid_prec)

  # --- CALCULATE THRESHOLDS FIRST ---
  best_idx_pr <- which.max(f1_score)
  best_threshold_pr <- sorted_probs[best_idx_pr]

  best_idx_roc <- which.max(youden_j)
  best_threshold_roc <- sorted_probs[best_idx_roc]

  # --- Accuracy at Best PR Threshold ---
  tn_at_best <- total_neg - fp_cum[best_idx_pr]
  accuracy_at_best <- (tp_cum[best_idx_pr] + tn_at_best) / (total_pos + total_neg)

  # --- Create Data Frame ---
  curve_data <- data.frame(
    Threshold = sorted_probs,
    TPR_Recall = tpr,
    FPR = fpr,
    Precision = precision,
    F1 = f1_score
  )

  # --- NEW: DOWNSAMPLING ---
  # If data is large, skip every 10 rows but guarantee best indices are kept
  n_rows <- nrow(curve_data)
  if (n_rows > 1000) {
    keep_idx <- unique(c(1, seq(1, n_rows, by = 10), best_idx_pr, best_idx_roc, n_rows))
    keep_idx <- sort(keep_idx) # Sort to keep the curve chronological
    curve_data <- curve_data[keep_idx, ]
  }

  # --- Create dedicated S3 objects for the individual curves ---
  roc_object <- list(
    curve = curve_data,
    auc = auc_roc,
    best_threshold = best_threshold_roc
  )
  class(roc_object) <- "quickSentiment_roc"

  prc_object <- list(
    curve = curve_data,
    auc = auc_pr,
    best_threshold = best_threshold_pr
  )
  class(prc_object) <- "quickSentiment_prc"

  # --- NEW: User's Requested Threshold Summary Table ---
  # Calculate Accuracy, Precision, Recall, and F1 at exact 0.1 increments
  summary_thresholds <- seq(0, 1, by = 0.1)
  summary_table <- data.frame(
    Threshold = summary_thresholds,
    Accuracy = NA, Precision = NA, Recall = NA, F1 = NA
  )

  for(i in 1:nrow(summary_table)) {
    th <- summary_table$Threshold[i]
    pred_pos <- predicted_probs >= th

    tp <- sum(pred_pos & is_positive)
    fp <- sum(pred_pos & !is_positive)
    tn <- sum(!pred_pos & !is_positive)
    fn <- sum(!pred_pos & is_positive)

    acc <- (tp + tn) / (total_pos + total_neg)
    prec <- ifelse((tp + fp) == 0, 0, tp / (tp + fp))
    rec <- ifelse((tp + fn) == 0, 0, tp / (tp + fn))
    f1 <- ifelse((prec + rec) == 0, 0, 2 * (prec * rec) / (prec + rec))

    summary_table$Accuracy[i] <- acc
    summary_table$Precision[i] <- prec
    summary_table$Recall[i] <- rec
    summary_table$F1[i] <- f1
  }

  # --- Package the Results (Flattened correctly!) ---
  results <- list(
    target_class       = positive_label,
    auc_roc            = auc_roc,
    best_threshold_roc = best_threshold_roc,
    auc_pr             = auc_pr,
    best_threshold_pr  = best_threshold_pr,
    accuracy_at_best   = accuracy_at_best,
    roc                = roc_object, # The ROC S3 Object
    prc                = prc_object,  # The PRC S3 Object
    threshold_summary  = summary_table
  )

  structure(results, class = "quickSentiment_eval")
}
