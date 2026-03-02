#' Train a Gradient Boosting Model using XGBoost
#'
#' This function trains a model using the xgboost package. It is highly
#' efficient and natively supports sparse matrices, making it ideal for text data.
#' It automatically handles both binary and multi-class classification problems.
#'
#' @param train_vectorized The training feature matrix (e.g., a `dfm` from quanteda).
#' @param Y The response variable for the training set. Should be a factor.
#' @param test_vectorized The test feature matrix, which must have the same
#'   features as `train_vectorized`.
#' @param parallel Logical
#' @param tune Logical
#'
#' @return A list containing four elements:
#'   \item{pred}{A vector of class predictions for the test set.}
#'   \item{probs}{A matrix of predicted probabilities.}
#'   \item{model}{The final, trained `xgb.Booster` model object.}
#'   \item{best_lambda}{Placeholder (NULL) for pipeline consistency.}
#'
#' @importFrom xgboost xgb.train xgb.DMatrix xgb.cv
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
#' # Provide column names (vocabulary) required by xgboost
#' colnames(train_matrix) <- paste0("word", 1:10)
#' colnames(test_matrix) <- paste0("word", 1:10)
#'
#' y_train <- factor(sample(c("P", "N"), 10, replace = TRUE))
#'
#' # Run xgboost model
#' model_results <- xgb_model(train_matrix, y_train, test_matrix)
#' }

xgb_model <- function(train_vectorized, Y, test_vectorized, parallel = FALSE,tune = FALSE) {

  message("\n--- Training XGBoost Model ---\n")

  # --- CHANGE 1: Prepare the Y variable ---
  # Get the number of classes from the factor levels
  num_classes <- length(levels(Y))
  # Convert factor Y ('neg', 'neu', 'pos') to numeric ('1','2','3') and then to zero-indexed ('0','1','2')
  y_train_numeric <- as.numeric(Y) - 1

  # Setup parallel threads
  threads <- if (isTRUE(parallel)) parallel::detectCores() - 1 else 1

  # --- MEMORY SAFETY: Cast quanteda DFM to dgCMatrix ---
  X_train_sparse <- methods::as(train_vectorized, "dgCMatrix")
  X_test_sparse <- methods::as(test_vectorized, "dgCMatrix")

  # This switch ensures Binary models allow Thresholds, while Multi-class works normally.
  if (num_classes == 2) {
    params <- list(
      objective = "binary:logistic",
      eval_metric = "logloss",
      eta = 0.1,
      max_depth = 4,
      nthread = threads
    )
  } else {
    params <- list(
      objective = "multi:softprob",
      eval_metric = "mlogloss",
      num_class = num_classes,
      eta = 0.1,
      max_depth = 4,
      nthread = threads
    )
  }

  # Prepare DMatrix objects (this part is the same)
  dtrain <- xgboost::xgb.DMatrix(data = X_train_sparse, label = y_train_numeric)
  dtest <- xgboost::xgb.DMatrix(data = X_test_sparse)

  # --- CONDITIONAL TUNING LOGIC ---
  if (isTRUE(tune)) {
    message("  - Using xgb.cv to find optimal nrounds (early stopping)...")

    cv_results <- xgboost::xgb.cv(
      params = params,
      data = dtrain,
      nfold = 5,
      nrounds = 500,
      early_stopping_rounds = 20,
      verbose = 0
    )

    best_nrounds <- cv_results$best_iteration

  } else {
    # Fast route: Use your original default
    best_nrounds <- 100
  }

  # Train the model (this part is the same)
  xgb_model <- xgboost::xgb.train(
    params = params,
    data = dtrain,
    nrounds = best_nrounds,
    verbose = 0
  )

  # --- CHANGE 3: Process the predictions ---
  # predict() now returns a matrix of probabilities (rows=samples, cols=classes)

  y_pred_prob_matrix <- predict(xgb_model, newdata = dtest, reshape = TRUE)

  if (num_classes == 2) {
    # Binary: Output is vector of probs for Class 1
    # Use default 0.5 threshold for the internal evaluation report
    prob_v <- as.vector(y_pred_prob_matrix)
    y_probs <- matrix(c(1 - prob_v, prob_v), ncol = 2)
    colnames(y_probs) <- levels(Y)

    y_pred_numeric <- ifelse(prob_v > 0.5, 1, 0)

    } else {
    # Multi-class: Output is Matrix
      y_probs <- y_pred_prob_matrix
      colnames(y_probs) <- levels(Y)
      y_pred_numeric <- max.col(y_probs) - 1
  }


  # Map the numeric predictions back to the original factor labels for evaluation
  y_pred_factor <- factor(y_pred_numeric, levels = 0:(num_classes-1), labels = levels(Y))
 
  # --- ENFORCE THE CONTRACT (this part is the same) ---
  results <- list(
    pred = y_pred_factor,
    probs = y_probs,
    model = xgb_model,
    best_lambda = NULL
  )

  return(results)
}
