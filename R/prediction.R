#' Predict Sentiment on New Data Using a Saved Pipeline Artifact
#'
#' This is a generic prediction function that handles different model types
#' and ensures consistent preprocessing and vectorization for new, unseen text.
#'
#' @param pipeline_object A list object returned by the main `pipeline()` function.
#'   It must contain the trained model, DFM template, preprocessing function,
#'   and n-gram settings.
#' @param df A data frame containing the new data.
#' @param text_column A string specifying the column name of the text to predict.
#'
#'
#' @return A vector of class predictions for the new data.
#'
#' @importFrom stats predict
#' @importFrom xgboost xgb.DMatrix
#'
#' @export
#' @examples
#' \donttest{
#' if (exists("my_artifacts")) {
#' preds <- prediction(my_artifacts, c("cleaned text one", "cleaned text two"))
#'  }
#' }
#'
prediction <- function(pipeline_object,
                       df,
                       text_column) {

  # --- 1. Extract the necessary artifacts from the pipeline object ---
  final_model <- pipeline_object$trained_model
  dfm_template <- pipeline_object$dfm_template
  class_levels <- pipeline_object$class_levels #for XGBoost since it does only numeric for Y

  # --- 2. Preprocess and vectorize the new data ---
  message("--- Preparing new data for prediction ---\n")

  new_dfm <- BOW_test(
    df[[text_column]],
    dfm_template)

  # --- 3. Conditional Prediction based on Model Class ---
  message("--- Making Predictions ---\n")

  # Initialize a variable to hold the final predictions
  final_predictions <- NULL

  # Check the class of the model object
  # Logistic Model
  if (inherits(final_model, "cv.glmnet") || inherits(final_model, "glmnet")) {
    s_val <- if (inherits(final_model, "cv.glmnet")) "lambda.min" else pipeline_object$best_lambda
    message("  - Detected a (cv)glmnet model. Predicting...\n")
    final_predictions <- stats::predict(final_model,
                                 newx = new_dfm,
                                 s = s_val,
                                 type = "class")
    # glmnet returns a matrix, so we extract the first column
    final_predictions <- final_predictions[, 1]
  # Naive Bayesian    
  } else if (inherits(final_model, "multinomial_naive_bayes")) {
    
    message("  - Detected a Naive Bayes model. Predicting...\n")
    final_predictions <- stats::predict(final_model, newdata = new_dfm)
  
  # Random Forest
  } else if (inherits(final_model, "ranger")) {

    message("  - Detected a ranger model. Predicting...\n")
    # Models like ranger/randomForest need a dense matrix
    new_dense_matrix <- as.matrix(new_dfm)

    message("Step 4: Correcting column names for ranger model...\n")
    model_colnames <- final_model$forest$independent.variable.names
    colnames(new_dense_matrix) <- model_colnames

    prediction_obj <- stats::predict(final_model, data = new_dense_matrix)
    final_predictions <- prediction_obj$predictions


  } else if (inherits(final_model, "xgb.Booster")) {
    message("  - Detected an xgboost model. Predicting...\n")
    # 1. Convert the new DFM into the required xgb.DMatrix format
    new_dmatrix <- xgb.DMatrix(data = new_dfm)

    # 2. Predict probabilities. reshape=TRUE creates a matrix for multi-class.
    pred_probabilities <- stats::predict(final_model, newdata = new_dmatrix, reshape = TRUE)

    # 3. Convert probabilities to final class labels
    if (is.vector(pred_probabilities)) { # Binary case
      # If the output is a single vector of probabilities for the positive class
      pred_numeric <- ifelse(pred_probabilities > 0.5, 1, 0)
    } else { # Multi-class case
      # If the output is a matrix, find the column with the highest probability for each row
      pred_numeric <- max.col(pred_probabilities) - 1 # Get 0-indexed class number
    }

    # 4. Convert numeric class (0, 1, 2...) back to the original factor levels
    final_predictions <- factor(pred_numeric,
                                levels = 0:(length(class_levels) - 1),
                                labels = class_levels)
  } else {
    stop(paste("Unsupported model type:", class(final_model)[1]))
  }

  message("--- Predictions Complete ---\n")
  return(final_predictions)
}

