#' Run a Full Text Classification Pipeline on Preprocessed Text
#'
#' This function takes a data frame with pre-cleaned text and handles the
#' data splitting, vectorization, model training, and evaluation.
#'
#' @param vect_method A string specifying the vectorization method.
#'   Defaults to \code{"bag_of_words"}.
#'   \itemize{
#'     \item \code{"bag_of_words"} (Alias: \code{"bow"}) - Standard count of words.
#'     \item \code{"term_frequency"} (Alias: \code{"tf"}) - Normalized counts.
#'     \item \code{"tfidf"} (Alias: \code{"tf-idf"}) - Term Frequency-Inverse Document Frequency.
#'     \item \code{"binary"} - Presence/Absence (1/0).
#'   }
#' @param model_name A string specifying the model to train.
#'   Defaults to \code{"logistic_regression"}.
#'   \itemize{
#'     \item \code{"random_forest"} (Alias: \code{"rf"})
#'     \item \code{"xgboost"} (Alias: \code{"xgb"})
#'     \item \code{"logistic_regression"} (Alias: \code{"logit"}, \code{"glm"})
#'   }
#' @param text_vector A character vector containing the **preprocessed** text.
#' @param sentiment_vector A vector or factor containing the target labels (e.g., ratings).
#' @param balance Logical. If TRUE, calculates inverse class weights to correct for imbalanced datasets. Defaults to FALSE.
#' @param n_gram The n-gram size to use for BoW/TF-IDF. Defaults to 1.
#' @param parallel If TRUE, runs model training in parallel. Default FALSE.
#' @param tune Logical. If TRUE, the pipeline will perform hyperparameter tuning
#'    for the selected model. Defaults to FALSE. [NEW]
#' @return A list containing the trained model object, the DFM template,
#'   class levels, and a comprehensive evaluation report.
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
#'
#'
#' out <- pipeline("bow", "naive_bayes",  text_vector = df$text, sentiment_vector = df$y)
#'
pipeline <- function(vect_method,
  model_name,
  text_vector,
  sentiment_vector,
  n_gram = 1,
  balance = FALSE,
  tune = FALSE,
  parallel=FALSE) {

    stopf <- function(...) stop(sprintf(...), call. = FALSE)

    # --- 1. CLEAN & TRANSLATE ARGUMENTS ---
    #to lower and trim ensures small typos don't break the function and allows for more flexible input
    vect_method <- tolower(trimws(vect_method))
    model_name  <- tolower(trimws(model_name))
    n_gram <- as.integer(n_gram)

    # ALIASING: VECTORIZERS
    if (vect_method == "bow") vect_method <- "bag_of_words"
    if (vect_method == "tf")  vect_method <- "term_frequency"
    if (vect_method == "tf-idf")  vect_method <- "tfidf"

    # ALIASING: Convert shortcuts to official names
    if (model_name == "rf")    model_name <- "random_forest"
    if (model_name == "xgb")   model_name <- "xgboost"
    if (model_name == "nb")   model_name <- "naive_bayes"
    if (model_name %in% c("logit","glm","logistic")) model_name <- "logistic_regression"


  # Ensure Vector method is in Input
    allowed_vect <- c("bag_of_words", "binary", "term_frequency", "tfidf")
    if (!vect_method %in% allowed_vect) {
      stopf("Vectorizer '%s' is not supported. Use: %s.",
            vect_method, paste(allowed_vect, collapse = ", "))
    }

  # Ensure Model name is in Input
    allowed_models <- c("logistic_regression", "random_forest", "xgboost", "naive_bayes")
    if (!model_name %in% allowed_models) {
      stopf("Model '%s' is not supported. Use: %s.",
            model_name, paste(allowed_models, collapse = ", "))
    }

  # Model list that can be used
    models_list <- list(
      logistic_regression = logit_model,
      random_forest       = rf_model,
      xgboost             = xgb_model,
      naive_bayes         = nb_model
    )


    if (!is.atomic(text_vector)) stopf("`text_vector` must be an atomic vector (usually character).")
    if (!is.atomic(sentiment_vector) && !is.factor(sentiment_vector)) stopf("`sentiment_vector` must be a vector or factor.")

    if (length(text_vector) != length(sentiment_vector)) {
      stopf("Length mismatch: `text_vector` has %d elements, but `label_vector` has %d elements.",
            length(text_vector), length(sentiment_vector))
    }

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

  #drop unnecessary columns and ensure sentiment is a factor
  initial_length <- length(text_vector)
  # Force formatting safely
  text_vector <- trimws(as.character(text_vector))
  text_vector[text_vector == ""] <- NA_character_
  sentiment_vector <- as.factor(sentiment_vector)

  # Find valid indices (where neither text nor label is NA)
  valid_idx <- !is.na(text_vector) & !is.na(sentiment_vector)

  # Subset vectors simultaneously
  text_vector <- text_vector[valid_idx]
  sentiment_vector <- sentiment_vector[valid_idx]

  rows_dropped <- initial_length - length(text_vector)

  if (rows_dropped > 0) {
    warning(sprintf("Dropped %d element(s) with missing/empty text or missing sentiments(labels).", rows_dropped),
            call. = FALSE)
  }
  if (length(text_vector) < 5) stopf("Not enough valid data after filtering (%d).", length(text_vector))
  if (nlevels(sentiment_vector) < 2) stopf("Need at least 2 sentiment classes after filtering.")


  # --- 2. TRAIN/TEST SPLIT ---

  # Base R Stratified Split
     train_idx <- unlist(lapply(split(seq_along(sentiment_vector), sentiment_vector), function(idx) {
    sample(idx, size = round(0.8 * length(idx)))
  }))

  # Ensure it's a clean numeric vector
  train_idx <- as.numeric(train_idx)

  text_train <- text_vector[train_idx]
  text_test <- text_vector[-train_idx]

  # Split Labels
  y_train <- sentiment_vector[train_idx]
  y_test  <- sentiment_vector[-train_idx]

  message(paste0("Data split: ", length(text_train), " training elements, ", length(text_test), " test elements.\n"))

  # --- 2.5. CLASS WEIGHTING (NEW) ---
  if (isTRUE(balance)) {

    # Get counts for each class in the training set
    class_counts <- table(y_train)
    total_obs <- length(y_train)
    num_classes <- length(class_counts)

    # Standard formula: Total Observations / (Number of Classes * Count in Class)
    weight_map <- total_obs / (num_classes * class_counts)

    # Map the correct weight to each individual observation in y_train
    obs_weights <- as.numeric(weight_map[as.character(y_train)])

  } else {
    obs_weights <- NULL
  }

  # --- 3. VECTORIZATION ---
  # This now operates on the pre-cleaned text column
  message(sprintf("Vectorizing with %s (ngram=%d)...", toupper(vect_method), n_gram))

  fit <- BOW_train(text_train,
                   weighting_scheme = vect_method,
                   ngram_size = n_gram)
  X_train <- fit$dfm_template

  X_test <- BOW_test(text_test,fit)



  # --- 4. MODEL TRAINING & PREDICTION ---

  model_results <- models_list[[model_name]](
    X_train,
    y_train,
    X_test,
    parallel = parallel,
    tune=tune,
    weights = obs_weights
    )
  if (is.null(model_results$model) || is.null(model_results$pred)) {
    stopf("Model function '%s' must return a list with elements `model` and `pred`.", model_name)
  }


  # --- 5. PACKAGE ARTIFACTS ---

 internal_type_map <- c(
  "logistic_regression" = "logit",
  "random_forest" = "rf",
  "xgboost" = "xgb",
  "naive_bayes" = "nb"
)
  # ---  RETURN FINAL RESULTS ---
  final_output <- list(
      trained_model = model_results$model,
      model_type    = internal_type_map[model_name],
      best_lambda   = model_results$best_lambda, # <- for glmnet only
      dfm_template = fit,
      class_levels = levels(y_test),
      ngram_size_used = n_gram,
      probs = model_results$probs,
      y_test= y_test
      )

  # --- 6. USER GUIDANCE MESSAGE ---
  baseline_accuracy <- mean(model_results$pred== y_test)
  acc_pct <- round(baseline_accuracy * 100, 2)

  message("\n======================================================")
  message(" --- quickSentiment Pipeline Complete ---")
  message(sprintf(" Model Type: %s", toupper(model_name)))
  message(sprintf(" Vectorizer: %s (ngram=%d)", toupper(vect_method), n_gram))
  message(sprintf(" Test Set Size: %d rows", length(y_test)))
  message(sprintf(" Accuracy of %s%% under baseline threshold.", acc_pct))
  message("======================================================\n")

  return(final_output)
}
