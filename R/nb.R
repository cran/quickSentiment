#' Multinomial Naive Bayes for Text Classification
#'
#' @param train_vectorized The training feature matrix (e.g., a `dfm` from quanteda).
#'   This should be a sparse matrix.
#' @param Y The response variable for the training set. Should be a factor for
#'   classification.
#' @param test_vectorized The test feature matrix, which must have the same
#'   features as `train_vectorized`
#' @param parallel Logical
#' @param tune Logical. If TRUE, tests different Laplace smoothing values.
#' @return A list containing four elements:
#'   \item{pred}{A vector of class predictions for the test set.}
#'   \item{probs}{A matrix of predicted probabilities.}
#'   \item{model}{The final, trained `naivebayes` model object.}
#'   \item{best_lambda}{Placeholder (NULL) for pipeline consistency.}
#' @importFrom naivebayes multinomial_naive_bayes
#' @importFrom stats predict
#' @importFrom methods as
#' @export
#' @examples
#' # 1. Create dummy numeric matrices with BOTH row and column names
#' train_matrix <- matrix(
#'   as.numeric(sample(0:5, 100, replace = TRUE)),
#'   nrow = 10, ncol = 10,
#'   dimnames = list(paste0("doc", 1:10), paste0("word", 1:10))
#' )
#'
#' test_matrix <- matrix(
#'   as.numeric(sample(0:5, 50, replace = TRUE)),
#'   nrow = 5, ncol = 10,
#'   dimnames = list(paste0("doc", 1:5), paste0("word", 1:10))
#' )
#'
#' # 2. Create dummy target variable
#' y_train <- factor(sample(c("P", "N"), 10, replace = TRUE))
#'
#' # 3. Run model
#' model_results <- nb_model(train_matrix, y_train, test_matrix)
#' print(model_results$pred)

nb_model <- function(train_vectorized, Y, test_vectorized, parallel = FALSE, tune = FALSE) {

  message("\n--- Training Multinomial Naive Bayes ---\n")

  best_laplace <- 1 # Default standard

  # --- MEMORY SAFETY: Cast quanteda DFM to dgCMatrix ---
  X_train_sparse <- methods::as(train_vectorized, "dgCMatrix")
  X_test_sparse <- methods::as(test_vectorized, "dgCMatrix")

  if (isTRUE(tune)) {
    message("  - Tuning: Performing 5-fold CV for Laplace smoothing...")

    # 1. Define Grid
    laplace_grid <- c(0.1, 0.5, 1.0, 1.5, 2.0)

    # 2. Create Folds (stratified)
    n_rows <- nrow(X_train_sparse)
    folds <- split(sample(seq_len(n_rows)), rep(1:5, length.out = n_rows))

    # Store errors for each laplace value
    grid_errors <- numeric(length(laplace_grid))

    # 3. Manual CV Loop (Keeps data sparse!)
    for (i in seq_along(laplace_grid)) {
      L <- laplace_grid[i]
      fold_errors <- numeric(length(folds))

      for (k in seq_along(folds)) {
        # Split data
        val_idx <- folds[[k]]

        # specific for sparse matrix slicing
        X_tr_fold <- X_train_sparse[-val_idx, , drop=FALSE]
        y_tr_fold <- Y[-val_idx]
        X_val_fold <- X_train_sparse[val_idx, , drop=FALSE]
        y_val_fold <- Y[val_idx]

        # Train & Predict
        fit <- naivebayes::multinomial_naive_bayes(x = X_tr_fold, y = y_tr_fold, laplace = L)
        preds <- stats::predict(fit, newdata = X_val_fold)

        # Calculate Error Rate
        fold_errors[k] <- mean(preds != y_val_fold)
      }
      grid_errors[i] <- mean(fold_errors)
    }

    # 4. Pick Winner
    best_idx <- which.min(grid_errors)
    best_laplace <- laplace_grid[best_idx]
  }

  # --- Final Training ---
  final_fit <- naivebayes::multinomial_naive_bayes(
    x = X_train_sparse,
    y = Y,
    laplace = best_laplace
  )

  y_probs <- stats::predict(final_fit, newdata = X_test_sparse, type = "prob")

  # EDGE CASE FIX: If test_vectorized has only 1 row, R might drop dimensions.
  # We force it back to a matrix to satisfy the pipeline contract.
  if (is.vector(y_probs)) {
    y_probs <- t(as.matrix(y_probs))
  }
  colnames(y_probs) <- levels(Y)
  y_pred <- stats::predict(final_fit, newdata = X_test_sparse) # Returns class labels by default

  # Ensure consistency
  y_pred_factor <- factor(y_pred, levels = levels(Y))

  return(list(
    pred = y_pred_factor,
    probs = y_probs,
    model = final_fit,
    best_lambda = NULL # Placeholder for pipeline consistency (NB doesn't use lambda)
  ))
}
