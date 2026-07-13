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
#' @param weights A numeric vector of observation weights. Default is NULL
#'
#' @return A list containing two elements:
#'   \item{pred}{A vector of class predictions for the test set.}
#'   \item{probs}{A matrix of predicted probabilities.}
#'   \item{model}{The final, trained `cv.glmnet` model object.}
#'   \item{best_lambda}{The optimal lambda value found during cross-validation.}
#'
#' @importFrom glmnet cv.glmnet
#' @importFrom stats predict
#' @importFrom doParallel registerDoParallel
#'
#' @export
#' @examples
#' \dontrun{
#' # Create dummy vectorized training and test data
#' train_matrix <- matrix(runif(100), nrow = 10, ncol = 10)
#' test_matrix <- matrix(runif(50), nrow = 5, ncol = 10)
#'
#' # Provide column names (vocabulary) required by glmnet
#' colnames(train_matrix) <- paste0("word", 1:10)
#' colnames(test_matrix) <- paste0("word", 1:10)
#'
#' y_train <- factor(sample(c("P", "N"), 10, replace = TRUE))
#'
#' # Run logistic regression model (glmnet)
#' model_results <- logit_model(train_matrix, y_train, test_matrix)
#' }
logit_model <- function(train_vectorized,
                        Y,
                        test_vectorized,
                        parallel=FALSE,
                        tune = FALSE,
                        weights = NULL){

  message("\n--- Running Logistic Regression (logit) Function ---\n")
  fam <- if (nlevels(Y) > 2) "multinomial" else "binomial"
  final_fit <- NULL
  best_s <- NULL

  if (isTRUE(tune)) {
    message("--- Tuning Elastic Net (alpha and lambda) ---\n")
    alpha_grid <- seq(0, 1, by = 0.2)
    best_cv_error <- Inf

    for (a in alpha_grid) {
      # cv.glmnet automatically handles lambda tuning and k-fold CV
      cv_fit <- cv.glmnet(
        x = train_vectorized,
        y = Y,
        family = fam,
        alpha = a,
        nfolds = 5,
        parallel = parallel,
        weights = weights
      )

    # Get the minimum error for this alpha
    min_err <- min(cv_fit$cvm)
    # If this alpha is better, save the model and parameters
    if (min_err < best_cv_error) {
      best_cv_error <- min_err
      final_fit <- cv_fit
      best_s <- cv_fit$lambda.min
      }

  }
  } else {
  final_fit <- cv.glmnet(
    x = train_vectorized,
    y = Y ,
    alpha = 1,           #  Lasso regularization
    nfolds = 5,
    family= fam,
    parallel = parallel,
    weights = weights
  )
  best_s <- final_fit$lambda.min
  }

   y_probs <- predict(final_fit,
                    newx = test_vectorized,
                    s = best_s,
                    type = "response")

  if (fam == "binomial") {
    # For binary, we use a standard 0.5 cutoff for the internal report
    p_class2 <- as.vector(y_probs)
    probs_matrix <- cbind(1 - p_class2, p_class2)
    colnames(probs_matrix) <- levels(Y)
    y_pred_idx <- ifelse(p_class2 > 0.5, 2, 1)
  } else {
    # For multi-class, we pick the column with the highest probability
    probs_matrix <- drop(y_probs)
    colnames(probs_matrix) <- levels(Y)

    # Prediction: Pick the column with max probability
    y_pred_idx <- max.col(probs_matrix)
  }
  y_pred_internal <- factor(levels(Y)[y_pred_idx], levels = levels(Y))

  results <- list(
    pred = y_pred_internal, #category
    probs = probs_matrix,
    model = final_fit,
    best_lambda = best_s
  )

  return(results)

}




