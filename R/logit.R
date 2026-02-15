#' Train a Regularized Logistic Regression Model using glmnet
#'
#' This function trains a logistic regression model using Lasso regularization
#' via the glmnet package. It uses cross-validation to automatically find the
#' optimal regularization strength (lambda).
#'
#' @param train_vectorized The training feature matrix (e.g., a `dfm` from quanteda).
#'   This should be a sparse matrix.
#' @param Y The response variable for the training set. Should be a factor for
#'   classification.
#' @param test_vectorized The test feature matrix, which must have the same
#'   features as `train_vectorized`.
#' @param parallel Logical
#' @param tune Logical
#'
#' @return A list containing two elements:
#'   \item{pred}{A vector of class predictions for the test set.}
#'   \item{model}{The final, trained `cv.glmnet` model object.}
#'
#' @importFrom glmnet cv.glmnet
#' @importFrom stats predict
#' @importFrom doParallel registerDoParallel
#'
#' @export
#' @examples
#' # Create dummy vectorized data
#' train_matrix <- matrix(runif(100), nrow = 10)
#' test_matrix <- matrix(runif(50), nrow = 5)
#' y_train <- factor(sample(c("P", "N"), 10, replace = TRUE))
#'
#' # Run model
#' model_results <- logit_model(train_matrix, y_train, test_matrix)
#' print(model_results$pred)
#'
logit_model <- function(train_vectorized, Y, test_vectorized, parallel=FALSE, tune = FALSE){
  message("\n--- Running Logistic Regression (logit) Function ---\n")
  if (isTRUE(tune)) {
    message("--- Tuning Elastic Net (alpha and lambda) ---\n")
    # Use caret for a grid search over alpha (Lasso/Ridge mix)
    tune_grid <- expand.grid(alpha = seq(0, 1, by = 0.2),
                             lambda = seq(0.001, 0.1, length.out = 10))
    # caret handles the cross-validation across the alpha grid
    trained_obj <- caret::train(x = train_vectorized, y = Y, method = "glmnet",
                          trControl = caret::trainControl(method = "cv", number = 5),
                          tuneGrid = tune_grid)

    # Extract the actual glmnet model and the best lambda
    final_fit <- trained_obj$finalModel
    best_s    <- trained_obj$bestTune$lambda

  } else {

  message("1. Training the glmnet model with 5-fold cross-validation...\n")
  final_fit <- cv.glmnet(
    x = train_vectorized,
    y = Y ,
    family = "multinomial",
    alpha = 1,           #  Lasso regularization
    nfolds = 5,
    parallel = parallel
  )
  best_s <- "lambda.min"
  }

   y_pred <- predict(final_fit,
                    newx = test_vectorized,
                    s = best_s,
                    type = "class")

  results <- list(
    pred = y_pred,
    model = final_fit
  )
  message("--- Logit function complete. Returning results. ---\n")
  return(results)

}




