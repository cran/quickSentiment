# quickSentiment 0.3.1

## Major Changes & API Updates
* `pipeline()` now accepts raw atomic vectors (`text_vector` and `sentiment_vector`) instead of full dataframes, drastically improving memory efficiency.
* Renamed the `prediction()` function to `predict_sentiment()` to prevent namespace collisions with base R generic functions.

## Performance & Dependency Improvements
* Removed the `caret` dependency entirely. Cross-validation folds and confusion matrix evaluations are now handled via lightweight custom implementations.
* The vectorization pipeline now strictly utilizes `dgCMatrix` sparse matrices, ensuring the package scales efficiently for large text datasets.

## New Features
* Models now return explicit class probabilities alongside the final predicted categories.
* Added automated ROC/AUC calculations for binary classification. The pipeline now returns an optimized threshold, which `predict_sentiment()` utilizes by default.
* Users can now manually define classification thresholds during inference.
* Upgraded text preprocessing capabilities with advanced arguments for custom stop words, retained words, and default corrections for `quanteda` stop word dictionaries.

## Bug Fixes
* Resolved an initialization bug that incorrectly prevented the use of the Naive Bayes (`nb`) model.
* Fixed a compatibility error that occurred when pairing TF-IDF vectorization with Random Forest (`rf`) models.
