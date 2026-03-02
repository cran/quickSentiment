#' Standard Negation Words for Sentiment Analysis
#'
#' A character vector of 25 common negation words. These words are automatically
#' protected by the \code{\link{pre_process}} function when \code{retain_negations = TRUE}
#' to prevent standard stopword lists from destroying sentiment polarity.
#'
#' @export
qs_negations <- c("no", "not", "nor", "neither", "never", "none", "cannot",
                  "aren't", "couldn't", "didn't", "doesn't", "don't",
                  "hadn't", "hasn't", "haven't", "isn't", "mightn't",
                  "mustn't", "needn't", "shan't", "shouldn't", "wasn't",
                  "weren't", "won't", "wouldn't")

#' Preprocess a Vector of Text Documents
#'
#' This function provides a comprehensive and configurable pipeline for cleaning
#' raw text data. It handles a variety of common preprocessing steps including
#' removing URLs and HTML, lowercasing, stopword removal, and lemmatization.
#'
#' @param doc_vector A character vector where each element is a document.
#' @param remove_brackets A logical value indicating whether to remove text in square brackets.
#' @param remove_urls A logical value indicating whether to remove URLs and email addresses.
#' @param remove_html A logical value indicating whether to remove HTML tags.
#' @param remove_nums A logical value indicating whether to remove numbers.
#' @param remove_emojis_flag A logical value indicating whether to remove common emojis.
#' @param to_lowercase A logical value indicating whether to convert text to lowercase.
#' @param remove_punct A logical value indicating whether to remove punctuation.
#' @param remove_stop_words A logical value indicating whether to remove English stopwords.
#' @param custom_stop_words A character vector of additional custom words to remove (e.g., c("rt", "via")). Default is NULL.
#' @param keep_words A character vector of words to protect from deletion (e.g., c("no", "not", "nor")). Default is NULL.
#' @param lemmatize A logical value indicating whether to lemmatize words to their dictionary form.
#' @param retain_negations Logical. If \code{TRUE} (the default), automatically protects common negation words (e.g., "not", "no", "never") from being deleted by the standard stopword list to preserve sentiment context.
#'
#' @return A character vector of the cleaned and preprocessed text.
#'
#' @importFrom stringr str_replace_all str_remove_all str_squish
#' @importFrom quanteda tokens tokens_select
#' @importFrom textstem lemmatize_strings
#' @importFrom stopwords stopwords
#'
#'
#' @export
#' @examples
#' raw_text <- c(
#'   "This is a <b>test</b>! Visit https://example.com",
#'   "Email me at test.user@example.org [important]"
#' )
#'
#' # Basic preprocessing with defaults
#' clean_text <- pre_process(raw_text)
#' print(clean_text)
#'
#' # Keep punctuation and stopwords
#' clean_text_no_stop <- pre_process(
#'   raw_text,
#'   remove_stop_words = FALSE,
#'   remove_punct = FALSE
#' )
#' print(clean_text_no_stop)
pre_process <- function(doc_vector,
  remove_brackets = TRUE,
  remove_urls = TRUE,
  remove_html = TRUE,
  remove_nums = FALSE,
  remove_emojis_flag = TRUE,
  to_lowercase = TRUE,
  remove_punct = TRUE,
  remove_stop_words = TRUE,
  custom_stop_words = NULL,
  keep_words = NULL,
  lemmatize = TRUE,
  retain_negations = TRUE) {

  # Validate input
  if (!is.atomic(doc_vector)) {
    stop("`doc_vector` must be an atomic vector (usually character).", call. = FALSE)
  }

  # Preserve NA positions and coerce safely
  doc_vector <- as.character(doc_vector)


  # --- Stage 1: String-level cleaning ---

  if (remove_brackets) {
    doc_vector <- stringr::str_replace_all(doc_vector, "\\[[^\\]]*\\]", " ")
  }
  if (remove_urls) {
    doc_vector <- stringr::str_replace_all(doc_vector, "(?i)\\bhttps?://\\S+\\b|\\bwww\\.\\S+\\b", "")
  }
  if (remove_html) {
    doc_vector <- stringr::str_replace_all(doc_vector, "<[^>]+>", "")
  }
  if (remove_emojis_flag) {
    doc_vector <- stringr::str_remove_all(doc_vector, "[\\U0001F600-\\U0001F64F]|[\\U0001F300-\\U0001F5FF]|[\\U0001F680-\\U0001F6FF]|[\\U0001F1E0-\\U0001F1FF]")
  }
  doc_vector <- stringr::str_squish(doc_vector)

  # --- Stage 2: Tokenization and token-level operations ---
  # A more efficient, standard workflow is to perform all string operations first.
  if (to_lowercase) {
    doc_vector <- base::tolower(doc_vector)
  }


  # Now, tokenize and remove stopwords/punctuation/numbers
  toks <- quanteda::tokens(doc_vector,
                          remove_punct = remove_punct,
                          remove_numbers = remove_nums)

  # --- Stopword Removal Logic ---
  if (remove_stop_words || !is.null(custom_stop_words)) {
    stops_to_remove <- if (remove_stop_words) quanteda::stopwords("en") else character(0)

    # 1. Handle Negations Explicitly
    if (retain_negations && remove_stop_words) {
      message("quickSentiment: Retaining negation words (e.g., 'not', 'no', 'never') to preserve sentiment polarity. To apply the strict stopword list instead, set `retain_negations = FALSE`. View qs_negations for more")


      # Protect negations and user's keep_words
      words_to_protect <- unique(c(tolower(keep_words), qs_negations))
      stops_to_remove <- setdiff(stops_to_remove, words_to_protect)

    } else {
      # If they turned it off, just protect their keep_words
      stops_to_remove <- setdiff(stops_to_remove, tolower(keep_words))
    }

    # 2. Add custom_stop_words LAST (The User's Absolute Override)
    if (!is.null(custom_stop_words)) {
      stops_to_remove <- unique(c(stops_to_remove, tolower(custom_stop_words)))
    }

    # 3. Filter the tokens
    toks <- quanteda::tokens_select(toks, pattern = stops_to_remove, selection = "remove")
  }


  # --- Final step: Convert tokens back to a single string ---
  out <- vapply(toks, function(x) paste(x, collapse = " "), character(1))
  if (lemmatize) {
    out <- textstem::lemmatize_strings(out)
  }
  out <- stringr::str_squish(out)
  return(out)


  }
