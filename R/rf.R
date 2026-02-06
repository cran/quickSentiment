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
rf_model <- function(train_vectorized, Y, test_vectorized,parallel = FALSE) {

  message("\n--- Training Random Forest Model (with ranger) ---\n")

  # --- Convert sparse DFM to a dense matrix ---
  X_train_matrix <- as.matrix(train_vectorized)
  X_test_matrix <- as.matrix(test_vectorized)

  threads <- if (isTRUE(parallel)) parallel::detectCores() - 1 else 1

  # --- Train the model ---
  train_df_for_ranger <- data.frame(Y, X_train_matrix, check.names = TRUE) # Being explicit

  ranger_model <- ranger(
    dependent.variable.name = "Y",
    data = train_df_for_ranger,
    num.trees = 100,
    num.threads = threads, # Control parallelism here
    verbose = FALSE
  )


  # 1. Get the exact predictor names the model was trained on.
  #    These are the "cleaned" names.
  training_colnames <- ranger_model$forest$independent.variable.names

  # 2. Force the test matrix to have those exact names.
  #    Our dfm_match() step already ensured the number and order of columns are the same.
  colnames(X_test_matrix) <- training_colnames

  # --- Make predictions ---
  predictions_obj <- predict(ranger_model, data = X_test_matrix)
  y_pred <- predictions_obj$predictions

  # --- ENFORCE THE CONTRACT ---
  results <- list(
    pred = y_pred,
    model = ranger_model
  )

  message("Ranger training complete.\n")

  return(results)
}
