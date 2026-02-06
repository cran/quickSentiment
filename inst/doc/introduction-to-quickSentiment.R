## ----include = FALSE----------------------------------------------------------
library(quickSentiment)


## ----setup--------------------------------------------------------------------
library(doParallel)
# CRAN limits the number of cores used during package checks
cores <- min(2, parallel::detectCores())
registerDoParallel(cores = cores)

## -----------------------------------------------------------------------------
# Look for the file in the installed package first
csv_path <- system.file("extdata", "tweets.csv", package = "quickSentiment")

# Fallback for when you are building the package locally
if (csv_path == "") {
  csv_path <- "../inst/extdata/tweets.csv"
}
tweets <- read.csv(csv_path)
set.seed(123)

## -----------------------------------------------------------------------------
tweets$cleaned_text <- pre_process(tweets$Tweet)
tweets$sentiment = ifelse(tweets$Avg>0,'P','N')

## -----------------------------------------------------------------------------
result <- pipeline(
  # --- Define the vectorization method ---
  # Options: "bow" (raw counts), "tf" (term frequency), "tfidf"
  vect_method = "tf",
  
  # --- Define the model to train ---
  # Options: "logit", "rf", "xgb"
  model_name = "rf",
  
  # --- Specify the data and column names ---
  df = tweets,
  text_column_name = "cleaned_text",      # The column with our preprocessed text
  sentiment_column_name = "sentiment",    # The column with the target variable
  
  # --- Set vectorization options ---
  # Use n_gram = 2 for unigrams + bigrams, or 1 for just unigrams
  n_gram = 1
)

## -----------------------------------------------------------------------------

tweets$sentimentPredict <- prediction(
  pipeline_object = result,
  df = tweets,
  text_column = "cleaned_text"
)


