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
  # Options: "bow" (raw counts), "tf" (term frequency), "tfidf", "binary"
  vect_method = "tf",
  
  # --- Define the model to train ---
  # Options: "logit", "rf", "xgb","nb"
  model_name = "rf",
  
  # --- Specify the data and column names ---
  text_vector = tweets$cleaned_text  ,   # The column with our preprocessed text
  sentiment_vector = tweets$sentiment,    # The column with the target variable
  
  # --- Set vectorization options ---
  # Use n_gram = 2 for unigrams + bigrams, or 1 for just unigrams
  n_gram = 1,
  parallel = cores
)

## -----------------------------------------------------------------------------

predicted_tweets <- predict_sentiment(
  pipeline_object = result,
  tweets$cleaned_text
)
head(predicted_tweets)

