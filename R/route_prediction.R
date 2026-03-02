#' Internal Router for Model Inference
#'
#' This is an internal helper function that routes a preprocessed sparse matrix
#' to the correct machine learning model. It guarantees that regardless of which
#' algorithm was trained, the output predictions and probability matrices follow
#' a strict, standardized format.
#'
#' @param new_data_vectorized A quanteda `dfm` containing the new text data.
#' @param qs_artifact The trained model list returned by the quickSentiment pipeline.
#' @param model_type A character string: "logit", "rf", "xgb", or "nb".
#' @param Y_levels A character vector of the original target variable levels.
#' @param threshold Numeric. The probability threshold for the positive class.
#'
#' @return A list containing:
#'   \item{pred}{A factor vector of class predictions.}
#'   \item{probs}{A properly named matrix of predicted probabilities.}
#'
#' @importFrom stats predict
#' @importFrom methods as
#' @importFrom xgboost xgb.DMatrix
#'
#' @noRd
route_prediction <- function(new_data_vectorized, qs_artifact, model_type, Y_levels, threshold) {

  # --- 1. GLOBAL MEMORY SAFETY ---
  # Force the DFM to a pure mathematical sparse matrix for all models
  X_new_sparse <- methods::as(new_data_vectorized, "dgCMatrix")
  num_classes <- length(Y_levels)

  # --- 2. MODEL ROUTING EXECUTION ---

  if (model_type == "rf") {

    expected_cols <- qs_artifact$trained_model$forest$independent.variable.names

    # 2. Handle total name wipe
    if (is.null(colnames(X_new_sparse)) && ncol(X_new_sparse) == length(expected_cols)) {
      colnames(X_new_sparse) <- expected_cols
    }

    current_cols <- colnames(X_new_sparse)
    if (!is.null(current_cols)) {
      missing_cols <- setdiff(expected_cols, current_cols)
      if (length(missing_cols) > 0) {
        # Build a sparse matrix of 0s for the missing words
        zero_padding <- Matrix::Matrix(0,
                                       nrow = nrow(X_new_sparse),
                                       ncol = length(missing_cols),
                                       sparse = TRUE)
        # Safely assign names to the padding
        dimnames(zero_padding) <- list(rownames(X_new_sparse), missing_cols)

        # Bind the missing columns back
        X_new_sparse <- cbind(X_new_sparse, zero_padding)

        # THE CRITICAL FIX: Force the names back on!
        colnames(X_new_sparse) <- c(current_cols, missing_cols)
      }
    } else {
      stop("Fatal error: New data matrix lost column names.")
    }

    # 3. Perfectly order the columns safely
    X_new_sparse <- X_new_sparse[, expected_cols, drop = FALSE]
    # ------------------------------------

    # Ranger returns a specific predictions object
    predictions_obj <- stats::predict(qs_artifact$trained_model, data = X_new_sparse)
    probs_matrix <- predictions_obj$predictions

  } else if (model_type == "xgb") {

    # XGBoost requires its own custom DMatrix format
    dtest <- xgboost::xgb.DMatrix(data = X_new_sparse)
    raw_probs <- stats::predict(qs_artifact$trained_model, newdata = dtest, reshape = TRUE)

    # XGB binary returns a single vector of positive probabilities
    if (num_classes == 2) {
      prob_v <- as.vector(raw_probs)
      probs_matrix <- matrix(c(1 - prob_v, prob_v), ncol = 2)
    } else {
      probs_matrix <- raw_probs
    }

  } else if (model_type == "logit") {

    # glmnet requires the saved best_lambda penalty parameter
    raw_probs <- stats::predict(qs_artifact$trained_model, newx = X_new_sparse,
                                type = "response", s = qs_artifact$best_lambda)

    # glmnet binary returns a single column of positive probabilities
    if (num_classes == 2) {
      prob_v <- as.vector(raw_probs)
      probs_matrix <- matrix(c(1 - prob_v, prob_v), ncol = 2)
    } else {
      # glmnet multiclass returns a 3D array (n_rows x n_classes x 1). Extract the 1st slice.
      probs_matrix <- raw_probs[, , 1]
    }

  } else if (model_type == "nb") {

    # Naive Bayes naturally returns a clean probability matrix
    probs_matrix <- stats::predict(qs_artifact$trained_model, newdata = X_new_sparse, type = "prob")

  } else {
    stop("Invalid model_type. Must be 'rf', 'xgb', 'logit', or 'nb'.")
  }

  # --- 3. EDGE CASE SAFETY NET ---
  # If the user only predicted ONE sentence, R drops dimensions to a vector.
  # We force it back into a 1-row, 2-column (or n-column) matrix.
  if (is.vector(probs_matrix) || length(dim(probs_matrix)) == 1) {
    probs_matrix <- t(as.matrix(probs_matrix))
  }

  # We apply the global standard to the column names so the wrapper can easily bind them
  colnames(probs_matrix) <- Y_levels

  # --- 4. CONVERT PROBABILITIES TO CLASSES ---
  if (ncol(probs_matrix) == 2) {
    # Binary: If Prob(Class2) > threshold, predict Class2
    y_pred_idx <- ifelse(probs_matrix[, 2] > threshold, 2, 1)
  } else {
    # Multiclass: Pick column with max probability (threshold is naturally ignored)
    y_pred_idx <- max.col(probs_matrix)
  }

  # Rebuild the final factor vector safely
  y_pred_factor <- factor(Y_levels[y_pred_idx], levels = Y_levels)

  return(list(
    pred = y_pred_factor,
    probs = probs_matrix
  ))
}
