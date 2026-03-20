#' Predict Sentiment on New Data Using a Saved Pipeline Artifact
#'
#' This is a generic prediction function that handles different model types
#' and ensures consistent preprocessing and vectorization for new, unseen text.
#'
#' @param pipeline_object A list object returned by the main `pipeline()` function.
#'   It must contain the trained model, DFM template, preprocessing function,
#'   and n-gram settings.
#' @param text_column A string specifying the column name of the text to predict.
#' @param threshold Numeric. Optional custom threshold for binary classification.
#'   If NULL, uses the optimized threshold from training (if available).
#'
#' @return A data frame containing the `predicted_class` and probability columns.
#'
#' @importFrom stats predict
#' @importFrom xgboost xgb.DMatrix
#'
#' @export
#' @examples
#' \donttest{
#' if (exists("my_artifacts")) {
#'   dummy_df <- data.frame(text = c("loved it", "hated it"), stringsAsFactors = FALSE)
#'   preds <- predict_sentiment(my_artifacts, df = dummy_df, text_column = "text")
#'  }
#' }
#'
predict_sentiment <- function(pipeline_object,
                       text_column,
                       threshold = 0.5) {

  # --- 1. Extract the necessary artifacts from the pipeline object ---
  final_model <- pipeline_object$trained_model
  dfm_template <- pipeline_object$dfm_template
  class_levels <- pipeline_object$class_levels #for XGBoost since it does only numeric for Y

  # --- 2. Preprocess and vectorize the new data ---
  message("--- Preparing new data for prediction ---\n")

  new_dfm <- BOW_test(
    text_column,
    dfm_template)

  # --- 3. Multi-Class Guardrail ---
  # If there are more than 2 classes, thresholds mathematically do not apply.
  if (length(class_levels) > 2) {
    if (threshold != 0.5) {
      warning("Threshold argument is ignored for multi-class prediction. Assigning class based on highest probability.")
    }
    threshold <- NULL # Pass NULL so route_prediction knows to just use max probability
  }

  # --- 4. Conditional Prediction based on Model Class ---
  message("--- Making Predictions ---\n")

  results <- route_prediction(
    new_data_vectorized = new_dfm,
    qs_artifact = pipeline_object,
    model_type = pipeline_object$model_type,
    Y_levels = class_levels,
    threshold = threshold
  )
  #--- 5.  Output
  final_output <- data.frame(predicted_class = results$pred)

  prob_df <- as.data.frame(results$probs)
  colnames(prob_df) <- paste0("prob_", colnames(prob_df))

  # Bind the probabilities to the class predictions
  final_output <- cbind(final_output, prob_df)

  message("--- Prediction Complete ---\n")
  return(final_output)

}
