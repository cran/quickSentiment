#' Run a Full Text Classification Pipeline on Preprocessed Text
#'
#' This function takes a data frame with pre-cleaned text and handles the
#' data splitting, vectorization, model training, and evaluation.
#'
#' @param vect_method A string specifying the vectorization method.
#'   Must be one of \code{"bow"}, \code{"binary"}, \code{"tf"}, or \code{"tfidf"}.
#' @param model_name A string specifying the model to train.
#'   Must be one of \code{"logit"}, \code{"rf"}, or \code{"xgb"}.
#' @param df The input data frame.
#' @param text_column_name The name of the column containing the **preprocessed** text.
#' @param sentiment_column_name The name of the column containing the original target labels (e.g., ratings).
#' @param n_gram The n-gram size to use for BoW/TF-IDF. Defaults to 1.
#' @param stratify If TRUE, use stratified split by sentiment. Default TRUE.
#' @param parallel If TRUE, runs model training in parallel. Default FALSE.
#'
#' @return A list containing the trained model object, the DFM template,
#'   class levels, and a comprehensive evaluation report.
#' @importFrom caret createDataPartition confusionMatrix
#' @importFrom parallel detectCores makeCluster stopCluster
#' @importFrom doParallel registerDoParallel
#' @importFrom foreach registerDoSEQ
#' @export
#' @examples
#' df <- data.frame(
#'   text = c("good product", "excellent", "loved it", "great quality",
#'            "bad service", "terrible", "hated it", "awful experience",
#'            "not good", "very bad", "fantastic", "wonderful"),
#'   y = c("P", "P", "P", "P", "N", "N", "N", "N", "N", "N", "P", "P")
#' )
#' # Note: We use a small dataset here for demonstration.
#' # In real use cases, ensure you have more observations per class.
#' out <- pipeline("bow", "logit", df, "text", "y")
#'
#'
pipeline <- function(vect_method,
  model_name,
  df,
  text_column_name,
  sentiment_column_name,
  n_gram = 1,
  parallel=FALSE,
  stratify = TRUE) {

    stopf <- function(...) stop(sprintf(...), call. = FALSE)
    vect_method <- tolower(trimws(vect_method))
    model_name  <- tolower(trimws(model_name))
    n_gram <- as.integer(n_gram)

  #Ensure Vector method is in Input
    if (!vect_method %in% c("bow", "binary", "tf", "tfidf")) {
      stopf("Vectorizer '%s' is not supported. Use: bow, binary, tf, tfidf.", vect_method)
    }

  #Ensure Model name is in Input
  models_list <- list(logit = logit_model, rf = rf_model, xgb = xgb_model)
    if (!model_name %in% names(models_list)) {
      stopf("Model '%s' is not supported. Use: %s.",
            model_name, paste(names(models_list), collapse = ", "))
    }


  if (!is.data.frame(df)) stopf("`df` must be a data.frame.")
  if (!text_column_name %in% names(df)) stopf("Column '%s' not found.", text_column_name)
  if (!sentiment_column_name %in% names(df)) stopf("Column '%s' not found.", sentiment_column_name)

  #FOR GLMNET ONLY
  if (isTRUE(parallel)) {
    # Dynamically detect cores and register
    n_cores <- parallel::detectCores() - 1
    cl <- parallel::makeCluster(n_cores)
    doParallel::registerDoParallel(cl)

    # Ensure the cluster stops even if the code crashes
    on.exit(parallel::stopCluster(cl), add = TRUE)
    on.exit(foreach::registerDoSEQ(), add = TRUE)
  }

  message(paste0("--- Running Pipeline: ", toupper(vect_method), " + ", toupper(model_name), " ---\n"))
  df$sentiment <- as.factor(df[[sentiment_column_name]])

  # --- 1. DATA CLEANING & PREP ---
  # Remove rows where the cleaned text is empty

  initial_rows <- nrow(df)
  df[[text_column_name]] <- trimws(as.character(df[[text_column_name]]))
  df[[text_column_name]][df[[text_column_name]] == ""] <- NA_character_

  df <- df[!is.na(df[[text_column_name]]) & !is.na(df$sentiment), , drop = FALSE]


  rows_dropped <- initial_rows - nrow(df)

  if (rows_dropped > 0) {
    warning(sprintf("Dropped %d row(s) with missing/empty text or missing labels.", rows_dropped),
            call. = FALSE)
  }

  if (nrow(df) < 5) stopf("Not enough rows after filtering (%d).", nrow(df))
  if (nlevels(df$sentiment) < 2) stopf("Need at least 2 sentiment classes after filtering.")

  # --- 2. TRAIN/TEST SPLIT ---


  if (isTRUE(stratify)) {
    train_idx <- caret::createDataPartition(df$sentiment, p = 0.8, list = FALSE)
  } else {
    train_idx <- sample(seq_len(nrow(df)), size = floor(0.8 * nrow(df)))
  }

  data_train <- df[train_idx, , drop = FALSE]
  data_test  <- df[-train_idx, , drop = FALSE]

  message(paste0("Data split: ", nrow(data_train), " training rows, ", nrow(data_test), " test rows.\n"))

  # --- 3. VECTORIZATION ---
  # This now operates on the pre-cleaned text column
  message(sprintf("Vectorizing with %s (ngram=%d)...", toupper(vect_method), n_gram))

  fit <- BOW_train(data_train[[text_column_name]],
                   weighting_scheme = vect_method,
                   ngram_size = n_gram)
  X_train <- fit$dfm_template

  X_test <- BOW_test(data_test[[text_column_name]],fit)

  # Prepare the response variables (y)
  y_train <- data_train$sentiment
  y_test <- data_test$sentiment

  # --- 4. MODEL TRAINING & PREDICTION ---

  model_results <- models_list[[model_name]](X_train, y_train, X_test, parallel = parallel)
  if (is.null(model_results$model) || is.null(model_results$pred)) {
    stopf("Model function '%s' must return a list with elements `model` and `pred`.", model_name)
  }


  # --- 5. EVALUATION ---

  predictions <- model_results$pred
  predictions_factor <- factor(predictions, levels = levels(y_test))

  evaluation <- caret::confusionMatrix(
  data = predictions_factor,
  reference = y_test,
  mode = "everything"
  )


  # --- 6. RETURN FINAL RESULTS ---
  final_output <- list(
  trained_model = model_results$model,
  dfm_template = fit,
  class_levels = levels(y_test),
  ngram_size_used = n_gram,
  vectorize_test_function = BOW_test,
  evaluation_report = evaluation
  )

  return(final_output)
  }
