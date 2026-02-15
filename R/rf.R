#' functions/random_forest_fast.R
#' Train a Random Forest Model using Ranger
#'
#' This function trains a Random Forest model using the high-performance ranger
#' package. It handles the necessary conversion from a sparse DFM to a dense
#' matrix and corrects for column name inconsistencies.
#'
#' @param train_vectorized The training feature matrix (e.g., a `dfm` from quanteda).
#' @param Y The response variable for the training set. Should be a factor.
#' @param test_vectorized The test feature matrix, which must have the same
#'   features as `train_vectorized`.
#' @param parallel Logical
#' @param tune Logical
#'
#'
#' @return A list containing two elements:
#'   \item{pred}{A vector of class predictions for the test set.}
#'   \item{model}{The final, trained `ranger` model object.}
#'
#' @importFrom ranger ranger
#' @importFrom stats predict
#'
#' @export
#' @examples
#' # Create dummy vectorized data
#' train_matrix <- matrix(runif(100), nrow = 10)
#' test_matrix <- matrix(runif(50), nrow = 5)
#' y_train <- factor(sample(c("P", "N"), 10, replace = TRUE))
#'
#' # Run model
#' model_results <- rf_model(train_matrix, y_train, test_matrix)
#' print(model_results$pred)
#'
rf_model <- function(train_vectorized, Y, test_vectorized, parallel = FALSE, tune = FALSE) {

  message("\n--- Training Random Forest Model (ranger) ---\n")

  p <- ncol(train_vectorized)
  threads <- if (isTRUE(parallel)) parallel::detectCores() - 1 else 1

  # Random Forest requires a dense matrix or a specific data frame format
  X_train_matrix <- as.matrix(train_vectorized)
  X_test_matrix <- as.matrix(test_vectorized)
  train_df <- data.frame(Y = Y, X_train_matrix, check.names = FALSE)

  if (isTRUE(tune)) {
    message("  - Tuning mtry using 5-fold cross-validation (this may take time)...")

    # Define a grid for mtry.
    # We test: sqrt(p), 2*sqrt(p), and p/3
    tune_grid <- expand.grid(
      mtry = unique(floor(c(sqrt(p), sqrt(p) * 2, p / 3))),
      splitrule = "gini",
      min.node.size = c(1, 5)
    )

    # We use caret here because ranger doesn't have a built-in cv function
    trained_obj <- caret::train(
      Y ~ .,
      data = train_df,
      method = "ranger",
      trControl = caret::trainControl(method = "cv", number = 5, allowParallel = parallel),
      tuneGrid = tune_grid,
      num.trees = 500
    )

    ranger_model <- trained_obj$finalModel
    message("    - Best mtry found: ", ranger_model$mtry)

  } else {
    # Fast Heuristic Route
    message("  - Using heuristic mtry and 500 trees...")
    actual_mtry <- min(p, floor(sqrt(p) * 2))

    ranger_model <- ranger::ranger(
      dependent.variable.name = "Y",
      data = train_df,
      num.trees = 500,
      mtry = actual_mtry,
      num.threads = threads,
      importance = "impurity",
      verbose = FALSE
    )
  }

  # Ensure test column names match training "clean" names
  colnames(X_test_matrix) <- ranger_model$forest$independent.variable.names
  predictions_obj <- predict(ranger_model, data = X_test_matrix)

  return(list(
    pred = predictions_obj$predictions,
    model = ranger_model
  ))
}
