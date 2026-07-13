# quickSentiment 0.3.5
## New Features
* Added the `balance` argument to `pipeline()`. When set to `TRUE`, the pipeline automatically calculates inverse class frequency weights to handle imbalanced datasets natively. These observation weights are passed directly into the loss functions of the underlying models (glmnet, ranger, and xgboost) without duplicating text data or artificially bloating the sparse matrix.
## Updates

* `evaluate_performance()` minor updates. Bug fixed


