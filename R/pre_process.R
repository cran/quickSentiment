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

#' Preprocess a Vector of Text Documents (Chunked, With Row-Level Fault Isolation)
#'
#' This function cleans raw text data using the same fast, vectorized string
#' and tokenization operations as before, but runs them in chunks. If a chunk
#' processes cleanly (the common case), it stays fully vectorized and fast.
#' If a chunk throws an error (e.g. from a malformed row), only THAT chunk
#' falls back to row-by-row processing so the exact offending row(s) can be
#' isolated, quarantined, and set to NA -- without slowing down or risking
#' the rest of the dataset.
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
#' @param retain_negations Logical. If \code{TRUE} (the default), automatically protects common negation words.
#' @param chunk_size Integer. Number of documents processed per vectorized batch.
#'   Defaults to 5000. Larger chunks are faster on clean data; smaller chunks
#'   isolate failures faster when a bad chunk needs to fall back to row-by-row.
#'
#' @return A character vector of cleaned text, same length as \code{doc_vector}.
#'   Rows that could not be processed are set to \code{NA_character_}. Details
#'   on any failures are attached as \code{attr(result, "quarantine")}, a
#'   data.frame with \code{row_index}, \code{original_text} (truncated to 200
#'   chars), and \code{error_message}.
#'
#' @importFrom stringr str_replace_all str_remove_all str_squish
#' @importFrom quanteda tokens tokens_select stopwords
#' @importFrom textstem lemmatize_strings
#'
#' @export
#' @examples
#' raw_text <- c(
#'   "This is a <b>test</b>! Visit https://example.com",
#'   "Email me at test.user@example.org [important]"
#' )
#' clean_text <- pre_process(raw_text)
#' print(clean_text)
#'
#' # Check for any quarantined rows after a run
#' attr(clean_text, "quarantine")
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
                        retain_negations = TRUE,
                        chunk_size = 5000) {

  # Validate input
  if (!is.atomic(doc_vector)) {
    stop("`doc_vector` must be an atomic vector (usually character).", call. = FALSE)
  }

  doc_vector <- as.character(doc_vector)
  n <- length(doc_vector)

  if (retain_negations && remove_stop_words) {
    message("quickSentiment: Retaining negation words (e.g., 'not', 'no', 'never') to preserve sentiment polarity. To apply the strict stopword list instead, set `retain_negations = FALSE`. View qs_negations for more")
  }

  # --- Precompute the stopword removal list ONCE, outside any loop ---
  # (This never changes per document, so recomputing it per row was pure waste.)
  stops_to_remove <- character(0)
  if (remove_stop_words || !is.null(custom_stop_words)) {
    stops_to_remove <- if (remove_stop_words) quanteda::stopwords("en") else character(0)

    if (retain_negations && remove_stop_words) {
      words_to_protect <- unique(c(tolower(keep_words), qs_negations))
      stops_to_remove <- setdiff(stops_to_remove, words_to_protect)
    } else {
      stops_to_remove <- setdiff(stops_to_remove, tolower(keep_words))
    }

    if (!is.null(custom_stop_words)) {
      stops_to_remove <- unique(c(stops_to_remove, tolower(custom_stop_words)))
    }
  }

  # --- Fast path: fully vectorized cleaning of a batch of documents ---
  # Used both for whole chunks (the common case) and, as a byproduct, for
  # single documents during row-level fallback below.
  clean_batch_vectorized <- function(batch) {
    if (remove_brackets) {
      batch <- stringr::str_replace_all(batch, "\\[[^\\]]*\\]", " ")
    }
    if (remove_urls) {
      batch <- stringr::str_replace_all(batch, "(?i)\\bhttps?://\\S+\\b|\\bwww\\.\\S+\\b", "")
    }
    if (remove_html) {
      batch <- stringr::str_replace_all(batch, "<[^>]+>", "")
    }
    if (remove_emojis_flag) {
      batch <- stringr::str_remove_all(batch, "[\\U0001F600-\\U0001F64F]|[\\U0001F300-\\U0001F5FF]|[\\U0001F680-\\U0001F6FF]|[\\U0001F1E0-\\U0001F1FF]")
    }
    batch <- stringr::str_squish(batch)

    if (to_lowercase) {
      batch <- base::tolower(batch)
    }

    toks <- quanteda::tokens(batch, remove_punct = remove_punct, remove_numbers = remove_nums)

    if (length(stops_to_remove) > 0) {
      toks <- quanteda::tokens_select(toks, pattern = stops_to_remove, selection = "remove")
    }

    out <- vapply(toks, function(x) paste(x, collapse = " "), character(1))
    if (lemmatize) {
      out <- textstem::lemmatize_strings(out)
    }
    stringr::str_squish(out)
  }

  cleaned_vector <- rep(NA_character_, n)
  quarantine_idx  <- integer(0)
  quarantine_text <- character(0)
  quarantine_err  <- character(0)

  starts <- seq(1, n, by = chunk_size)
  message(sprintf("  - Preprocessing %d document(s) in %d chunk(s) of up to %d...",
                  n, length(starts), chunk_size))

  for (start in starts) {
    end <- min(start + chunk_size - 1, n)
    idx <- start:end
    chunk <- doc_vector[idx]

    chunk_result <- tryCatch(
      list(ok = TRUE, value = clean_batch_vectorized(chunk)),
      error = function(e) list(ok = FALSE, error = conditionMessage(e))
    )

    # Happy path: the whole chunk processed fine and shapes match -- stay fast.
    if (chunk_result$ok && length(chunk_result$value) == length(idx)) {
      cleaned_vector[idx] <- chunk_result$value
      next
    }

    # --- Fallback: this chunk failed (or produced a mismatched shape) ---
    # Only THIS chunk pays the row-by-row cost; every other chunk stays fast.
    message(sprintf(
      "  - Rows %d-%d needed row-by-row isolation (%s). Retrying individually...",
      start, end,
      if (chunk_result$ok) "output length mismatch" else chunk_result$error
    ))

    for (i in seq_along(idx)) {
      row_idx  <- idx[i]
      row_text <- chunk[i]

      if (is.na(row_text)) {
        cleaned_vector[row_idx] <- NA_character_
        next
      }

      row_result <- tryCatch(
        list(ok = TRUE, value = clean_batch_vectorized(row_text)),
        error = function(e) list(ok = FALSE, error = conditionMessage(e))
      )

      if (row_result$ok && length(row_result$value) == 1) {
        cleaned_vector[row_idx] <- row_result$value
      } else {
        cleaned_vector[row_idx] <- NA_character_
        quarantine_idx  <- c(quarantine_idx, row_idx)
        quarantine_text <- c(quarantine_text, substr(row_text, 1, 200))
        quarantine_err  <- c(quarantine_err,
                             if (row_result$ok) "unexpected output shape" else row_result$error)
      }
    }
  }

  if (length(quarantine_idx) > 0) {
    warning(sprintf(
      "pre_process: %d row(s) could not be processed and were set to NA. Inspect attr(result, 'quarantine') for details.",
      length(quarantine_idx)
    ), call. = FALSE)
  }

  attr(cleaned_vector, "quarantine") <- data.frame(
    row_index     = quarantine_idx,
    original_text = quarantine_text,
    error_message = quarantine_err,
    stringsAsFactors = FALSE
  )

  cleaned_vector
}
