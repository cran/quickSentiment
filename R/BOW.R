#BOW.R
#' Train a Bag-of-Words Model
#' @param doc A character vector of documents to be processed.
#' @param weighting_scheme A string specifying the weighting to apply.
#'   Defaults to \code{"bag_of_words"}.
#'   \itemize{
#'     \item \code{"bag_of_words"} (Alias: \code{"bow"}) - Standard count of words.
#'     \item \code{"term_frequency"} (Alias: \code{"tf"}) - Normalized counts (frequency relative to document length).
#'     \item \code{"tfidf"} (Alias: \code{"tf-idf"}) - Term Frequency-Inverse Document Frequency.
#'     \item \code{"binary"} - Presence/Absence (1/0).
#'   }
#' @param ngram_size An integer specifying the maximum n-gram size. For example,
#' `ngram_size = 1` will create unigrams only; `ngram_size = 2` will create unigrams and bigrams. Defaults to 1.
#' @return An object of class \code{"qs_bow_fit"} containing:
#'   \itemize{
#'     \item \code{dfm_template}: a quanteda \code{dfm} template
#'     \item \code{weighting_scheme}: the weighting used
#'     \item \code{ngram_size}: the n-gram size used
#'   }#'
#' @importFrom quanteda tokens tokens_ngrams tokens_select dfm dfm_weight ndoc docfreq dfm_tfidf
#' @importFrom magrittr %>%
#' @importFrom stopwords stopwords
#' @export
#' @examples
#' txt <- c("text one", "text two text")
#' fit <- BOW_train(txt, weighting_scheme = "bow")
#' fit$dfm_template
#'
BOW_train <- function(doc,weighting_scheme = "bow",ngram_size=1) {
  idf_vector <- NULL

  #check for proper inputs
  weighting_scheme <- tolower(trimws(weighting_scheme))
  if (weighting_scheme == "bow") weighting_scheme <- "bag_of_words"
  if (weighting_scheme == "tf")  weighting_scheme <- "term_frequency"
  if (weighting_scheme == "tf-idf")  weighting_scheme <- "tfidf"

  allowed <- c("bag_of_words", "binary", "term_frequency", "tfidf")

  if (!weighting_scheme %in% allowed) {
    stop(sprintf("Invalid weighting_scheme '%s'. Must be one of: %s",
                 weighting_scheme, paste(allowed, collapse = ", ")),
         call. = FALSE)
  }

  #N-Gram check
  ngram_size <- as.integer(ngram_size)
  if (ngram_size < 1) {
    stop("`ngram_size` must be >= 1.", call. = FALSE)
  }

  message(paste0("  - Fitting BoW model (", weighting_scheme, ") on training data..."))

  # Create the DFM
  dfm_raw <- doc %>%
    quanteda::tokens() %>%
    quanteda::tokens_ngrams(n = 1:ngram_size) %>%
    quanteda::dfm()

  # Store IDF vector separately ONLY if using tfidf for manual matching in test
  # Note: quanteda's dfm_tfidf is preferred, but for BOW_test to apply training
  # weights to new data, we still need the training IDF values.
  idf_vals <- NULL

  if (weighting_scheme == "tfidf") {
    # Using the standard smoothed IDF formula as you had before
    N <- quanteda::ndoc(dfm_raw)
    df_counts <- quanteda::docfreq(dfm_raw)
    idf_vals <- log((N + 1) / (df_counts + 1)) + 1
  }
  # Weighting Logic
  dfm_final <- switch(weighting_scheme,
                      "binary" = quanteda::dfm_weight(dfm_raw, scheme = "boolean"),
                      "term_frequency"     = quanteda::dfm_weight(dfm_raw, scheme = "prop"),
                      "tfidf"  = quanteda::dfm_tfidf(dfm_raw, scheme_tf = "prop", scheme_df = "inverse",base=exp(1)), #force natural log,
                      dfm_raw # default bow
  )

  #returns not just the dfm but also the weighting scheme and ngram size for future reference

  fit <- list(
    dfm_template = dfm_final,
    weighting_scheme = weighting_scheme,
    ngram_size = ngram_size,
    idf_vector = if (weighting_scheme == "tfidf") idf_vals else NULL
  )
  class(fit) <- "qs_bow_fit"
  return(fit)
}
#' Transform New Text into a Document-Feature Matrix
#'
#' This function takes a character vector of new documents and transforms it
#' into a DFM that has the exact same features as a pre-fitted training DFM,
#' ensuring consistency for prediction.
#'
#' @param doc A character vector of new documents to be processed.
#' @param fit A fitted BoW object returned by \code{BOW_train()}.
#' @return A quanteda \code{dfm} aligned to the training features.
#'
#' @importFrom quanteda tokens tokens_ngrams tokens_select dfm dfm_match featnames dfm_weight
#' @importFrom magrittr %>%
#' @importFrom Matrix Diagonal
#' @export
#' @examples
#' train_txt <- c("apple orange banana", "apple apple")
#' fit <- BOW_train(train_txt, weighting_scheme = "bow")
#' new_txt <- c("banana pear", "orange apple")
#' test_dfm <- BOW_test(new_txt, fit)
#' test_dfm

#  BOW_test function

BOW_test <- function(doc, fit) {
  # check for proper inputs
  if (!inherits(fit, "qs_bow_fit")) {
    stop("`fit` must be an object of class 'qs_bow_fit'.", call. = FALSE)
  }

  if (is.null(fit$dfm_template) || is.null(fit$weighting_scheme) || is.null(fit$ngram_size)) {
    stop("`fit` is missing required components.", call. = FALSE)
  }

  # Extract artifacts
  training_dfm <- fit$dfm_template
  weighting_scheme <- fit$weighting_scheme
  ngram_size <- fit$ngram_size
  saved_idf <- fit$idf_vector

  message(paste0("  - Applying BoW transformation (", weighting_scheme, ") to new data..."))

  # Create a raw DFM and match its features to the training template
  dfm_raw <- doc %>%
    quanteda::tokens() %>%
    quanteda::tokens_ngrams(n = 1:ngram_size)%>%
    quanteda::dfm()

  # Align features to training template
  dfm_matched <- quanteda::dfm_match(dfm_raw, features = quanteda::featnames(training_dfm))

  # 3. Apply the SAME weighting scheme
  dfm_final <- switch(fit$weighting_scheme,
                      "binary" = quanteda::dfm_weight(dfm_matched, scheme = "boolean"),
                      "term_frequency"     = quanteda::dfm_weight(dfm_matched, scheme = "prop"),
                      "tfidf"  = {
                        # Use the specific IDF vector calculated during training
                        tf_test <- quanteda::dfm_weight(dfm_matched, scheme = "prop")
                        weighted_mat <- tf_test %*% Matrix::Diagonal(x = fit$idf_vector)
                        quanteda::as.dfm(weighted_mat)
                      },
                      dfm_matched # Default for "bag of words" (raw counts)
  )
  return(dfm_final)
}
