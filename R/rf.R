#' functions/random_forest_fast.R
#' Train a Random Forest Model using Ranger
#'
#' This function trains a Random Forest model using the high-performance ranger
#' package. It natively utilizes sparse matrices (dgCMatrix) to avoid memory
#' exhaustion and utilizes Out-Of-Bag (OOB) error for rapid hyperparameter tuning.
#' @param train_vectorized The training feature matrix (e.g., a `dfm` from quanteda).
#' @param Y The response variable for the training set. Should be a factor.
#' @param test_vectorized The test feature matrix, which must have the same
#'   features as `train_vectorized`.
#' @param parallel Logical
#' @param tune Logical. If TRUE, tunes `mtry` using native OOB error
#' @param weights A numeric vector of observation weights. Default is NULL
#'
#' @return A list containing four elements:
#'   \item{pred}{A vector of class predictions for the test set.}
#'   \item{probs}{A matrix of predicted probabilities.}
#'   \item{model}{The final, trained `ranger` model object.}
#'   \item{best_lambda}{Placeholder (NULL) for pipeline consistency.}
#'
#' @importFrom ranger ranger
#' @importFrom stats predict
#' @importFrom methods as
#'
#' @export
#' @examples
#' \dontrun{
#' # Create dummy vectorized training and test data
#' train_matrix <- matrix(runif(100), nrow = 10, ncol = 10)
#' test_matrix <- matrix(runif(50), nrow = 5, ncol = 10)
#'
#' # Provide column names (vocabulary) required by ranger
#' colnames(train_matrix) <- paste0("word", 1:10)
#' colnames(test_matrix) <- paste0("word", 1:10)
#'
#' y_train <- factor(sample(c("P", "N"), 10, replace = TRUE))
#'
#' # Run random forest model
#' model_results <- rf_model(train_matrix, y_train, test_matrix)
#' }
rf_model <- function(train_vectorized,
                     Y,
                     test_vectorized,
                     parallel = FALSE,
                     tune = FALSE,
                     weights = NULL) {

  message("\n--- Training Random Forest Model (ranger) ---\n")

  p <- ncol(train_vectorized)
  threads <- if (isTRUE(parallel)) parallel::detectCores() - 1 else 1

  # Check if anyone passed a dataframe
  if (is.data.frame(train_vectorized)) {
    train_vectorized <- as.matrix(train_vectorized)
    test_vectorized <- as.matrix(test_vectorized)
  }
  # Cast the quanteda DFM directly to a standard dgCMatrix
  X_train_matrix<- methods::as(train_vectorized, "dgCMatrix")
  X_test_matrix <- methods::as(test_vectorized, "dgCMatrix")

  if (isTRUE(tune)) {
    message("  - Tuning mtry using native Out-Of-Bag (OOB) error(this may take time)...")

    # Define a grid for mtry.
    # We test: sqrt(p), 2*sqrt(p), and p/3
    # Define a smart grid for mtry based on feature count (p)
    mtry_grid <- unique(floor(c(sqrt(p), sqrt(p) * 2, p / 3)))
    best_oob_error <- Inf
    ranger_model <- NULL

    for (m in mtry_grid) {
      # Train a model directly on the sparse matrices using x/y
      temp_model <- ranger::ranger(
        x = X_train_matrix,
        y = Y,
        num.trees = 200,
        mtry = m,
        num.threads = threads,
        importance = "impurity",
        probability = TRUE,   # We need probabilities for ROC/AUC later
        verbose = FALSE,
        case.weights = weights
      )
     # ranger saves the OOB error automatically!
      # (Since probability=TRUE, this is the Brier score)
      current_error <- temp_model$prediction.error
      # Keep the model if it has the lowest error so far
      if (current_error < best_oob_error) {
        best_oob_error <- current_error
        ranger_model <- temp_model
      }
    }

  } else {
    # Fast Heuristic Route

    actual_mtry <- min(p, floor(sqrt(p) * 2))

    ranger_model <- ranger::ranger(
      x = X_train_matrix,
      y = Y,
      num.trees = 200,
      mtry = actual_mtry,
      num.threads = threads,
      importance = "impurity",
      verbose = FALSE,
      probability = TRUE,
      case.weights = weights
    )
  }

  #Required for TF-IDF
  expected_cols <- colnames(X_train_matrix)

  # Handle total name wipe
  if (is.null(colnames(X_test_matrix)) && ncol(X_test_matrix) == length(expected_cols)) {
    colnames(X_test_matrix) <- expected_cols
  }

  current_cols <- colnames(X_test_matrix)
  if (!is.null(current_cols)) {
    missing_cols <- setdiff(expected_cols, current_cols)
    if (length(missing_cols) > 0) {
      # Build a sparse matrix of 0s for the missing words
      zero_padding <- Matrix::Matrix(0,
                                     nrow = nrow(X_test_matrix),
                                     ncol = length(missing_cols),
                                     sparse = TRUE)
      # Safely assign names to the padding
      dimnames(zero_padding) <- list(rownames(X_test_matrix), missing_cols)

      # Bind the missing columns back
      X_test_matrix <- cbind(X_test_matrix, zero_padding)

      # THE CRITICAL FIX: Force the names back on!
      # (Prevents cbind from stripping names)
      colnames(X_test_matrix) <- c(current_cols, missing_cols)
    }
  } else {
    stop("Fatal error: Test matrix lost column names and dimensions do not match training data.")
  }

   X_test_matrix <- X_test_matrix[, expected_cols, drop = FALSE]


  predictions_obj <- predict(ranger_model, data = X_test_matrix)
  probs_matrix <- predictions_obj$predictions

  # CONVERT TO CLASSES (For Pipeline Internal Report) ---
  if (ncol(probs_matrix) == 2) {
    # Binary: If Prob(Class2) > 0.5, predict Class2
    y_pred_idx <- ifelse(probs_matrix[, 2] > 0.5, 2, 1)
  } else {
    # Multiclass: Pick column with max probability
    y_pred_idx <- max.col(probs_matrix)
  }

  # Convert indices back to original Factor Levels
  y_pred_factor <- factor(levels(Y)[y_pred_idx], levels = levels(Y))
  message("--- Random Forest complete. Returning results. ---\n")
  return(list(
    pred = y_pred_factor,
    probs = probs_matrix,
    model = ranger_model,
    best_lambda = NULL
  ))
}
