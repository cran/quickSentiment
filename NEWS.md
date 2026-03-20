# quickSentiment 0.3.2

## Major Changes & API Updates
* `pipeline()` no longer computes ROC or returns ROC guidelines
* `predict_sentiment()` no longer inherits a threshold directly from the pipeline. It now defaults to a standard 0.5 classification threshold, while still allowing users to manually pass the custom cutoff they feel most comfortable with.
* `evaluate_performance()` is now available. Users can now pass model predictions and test data to calculate detailed metrics (including AUC, Precision, and Recall) targeted at any specific factor level or class of interest.

## Performance & Dependency Improvements
* `pipeline()` does less computing then before

## New Features
* The new evaluate_performance() engine leverages R's S3 object-oriented system. It returns custom evaluation objects that natively integrate with base R's plot() function to instantly generate high-quality ROC and Precision-Recall curves.
