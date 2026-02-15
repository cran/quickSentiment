#' Train a Naive Bayes Model
#' @param train_vectorized The training feature matrix (e.g., a `dfm` from quanteda).
#'   This should be a sparse matrix.
#' @param Y The response variable for the training set. Should be a factor for
#'   classification.
#' @param test_vectorized The test feature matrix, which must have the same
#'   features as `train_vectorized`
#' @param parallel Logical
#' @param tune Logical. If TRUE, tests different Laplace smoothing values.
#' @importFrom naivebayes multinomial_naive_bayes
#' @importFrom caret train trainControl
#' @importFrom stats predict
#' @export
#' @examples
#' #Create dummy vectorized data
#' train_matrix <- matrix(runif(100), nrow = 10)
#' test_matrix <- matrix(runif(50), nrow = 5)
#' colnames(train_matrix) <- paste0("word", 1:10)
#' colnames(test_matrix) <- paste0("word", 1:10)
#' y_train <- factor(sample(c("P", "N"), 10, replace = TRUE))
#' # Run model
#' model_results <- nb_model(train_matrix, y_train, test_matrix)
#' print(model_results$pred)

nb_model <- function(train_vectorized, Y, test_vectorized, parallel = FALSE, tune = FALSE) {
  
  message("\n--- Training Sparse Multinomial Naive Bayes ---\n")
  
  if (isTRUE(tune)) {
    message("  - Tuning Laplace smoothing value using 5-fold CV...")
    
   
    # 0 is no smoothing, 1 is standard, 2 is heavy smoothing
    tune_grid <- expand.grid(laplace = c(0, 0.5, 1, 1.5, 2))
    
    
    # Note: multinomial_naive_bayes is very fast even in a loop
    trained_obj <- caret::train(
      x = train_vectorized, 
      y = Y, 
      method = "multinomial_naive_bayes",
      trControl = caret::trainControl(method = "cv", number = 5),
      tuneGrid = tune_grid
    )
    
    nb_fit <- trained_obj$finalModel
    best_laplace <- trained_obj$bestTune$laplace
    message("    - Best Laplace found: ", best_laplace)
    
  } else {
    # Fast route: Use the standard Laplace = 1
    message("  - Using default Laplace smoothing (1)")
    nb_fit <- naivebayes::multinomial_naive_bayes(
      x = train_vectorized, 
      y = Y, 
      laplace = 1
    )
  }
  
  y_pred <- predict(nb_fit, newdata = test_vectorized)
  
  return(list(
    pred = y_pred,
    model = nb_fit
  ))
}