#!/usr/bin/env Rscript

# Train a PROP model on the packaged training features and classify feature
# vectors. Predictions are printed to the console only; no files are written.
#
# Usage:
#   Rscript scripts/run_prop_classify.R --input=new_features.csv
#   Rscript scripts/run_prop_classify.R --input=new_features.csv --labels=my_labels.csv
#   Rscript scripts/run_prop_classify.R --input=new_features.csv --train-ratio=0.7
#   Rscript scripts/run_prop_classify.R --input=new_features.csv --screening=l1lr --transform=zscore --seed=2026
#   Rscript scripts/run_prop_classify.R            # classify the packaged training features themselves
#
# Input CSV format (same schema as data/features_resnet50.csv):
#   ID, filename, f0001, f0002, ..., f2048
# ID/filename columns are optional; the 2048 f-columns are required.
#
# Reusable hook (source the script instead of running it):
#   source("scripts/run_prop_classify.R")
#   d <- prepare_prop_data(input = "new_features.csv", train_ratio = 0.7)
#   X_train <- d$X_train; y_train <- d$y_train; X_test <- d$X_test; X_new <- d$X_new
#   # then call prop_train() / prop_predict() with your own split

# Locate the repository root (works when run via Rscript or sourced from the repo).
find_repo_root <- function() {
  script_flag <- grep("^--file=", commandArgs(FALSE), value = TRUE)
  if (length(script_flag) > 0) {
    script_path <- normalizePath(sub("^--file=", "", script_flag[[1]]), winslash = "/")
    normalizePath(file.path(dirname(script_path), ".."), winslash = "/")
  } else {
    normalizePath(getwd(), winslash = "/")
  }
}

source(file.path(find_repo_root(), "src", "PROP.R"))

# Stratified split: within each class take floor(class_size * train_ratio) as
# training, the rest as test.
split_prop_data <- function(y, train_ratio) {
  class_1 <- which(y == 1L)
  class_2 <- which(y == 2L)
  n_1 <- floor(length(class_1) * train_ratio)
  n_2 <- floor(length(class_2) * train_ratio)
  stopifnot(n_1 >= 1L, n_2 >= 1L)
  train <- c(sample(class_1, n_1), sample(class_2, n_2))
  list(train = train, test = setdiff(seq_along(y), train))
}

# Prepare the packaged training data (+ optional input CSV) as R objects.
# train_ratio: NULL (default) keeps all data as training; a value in (0, 1)
#   stratifies the packaged data into train/test by that fraction.
# Returns a list with X_train, y_train, X_test, y_test, X_new, id_col, fn_col,
# test_id_col, test_fn_col, feature_columns.
prepare_prop_data <- function(input = NULL, train_ratio = NULL,
                              seed = 2026L,
                              labels = NULL,
                              repo_root = find_repo_root()) {
  repo_root <- normalizePath(repo_root, winslash = "/")
  features_csv <- file.path(repo_root, "data", "features_resnet50.csv")
  if (is.null(labels)) labels <- file.path(repo_root, "data", "cataract_labels.csv")
  labels_csv <- labels
  feat_dt <- data.table::fread(features_csv)
  if (!file.exists(labels_csv)) {
    stop("Training labels file not found: ", labels_csv,
         "\nProvide it via --labels (or restore data/cataract_labels.csv).")
  }
  lab_dt <- data.table::fread(labels_csv)
  stopifnot(identical(feat_dt$ID, lab_dt$ID), identical(feat_dt$filename, lab_dt$filename))

  feature_columns <- grep("^f[0-9]+$", names(feat_dt), value = TRUE)
  stopifnot(length(feature_columns) == 2048L, all(lab_dt$Y_cataract %in% c(0L, 1L)))
  X_all <- as.matrix(feat_dt[, ..feature_columns])
  y_all <- as.integer(lab_dt$Y_cataract) + 1L  # 0/1 -> 1/2

  if (!is.null(train_ratio)) {
    stopifnot(is.numeric(train_ratio), length(train_ratio) == 1L,
              train_ratio > 0, train_ratio < 1)
    set.seed(seed)
    split <- split_prop_data(y_all, train_ratio)
    X_train <- X_all[split$train, , drop = FALSE]
    y_train <- y_all[split$train]
    X_test <- X_all[split$test, , drop = FALSE]
    y_test <- y_all[split$test]
    test_id_col <- as.character(feat_dt$ID)[split$test]
    test_fn_col <- as.character(feat_dt$filename)[split$test]
  } else {
    X_train <- X_all
    y_train <- y_all
    X_test <- NULL
    y_test <- NULL
    test_id_col <- NULL
    test_fn_col <- NULL
  }

  if (!is.null(input)) {
    if (!file.exists(input)) stop("Input file not found: ", input)
    new_dt <- data.table::fread(input)
    new_feature_columns <- grep("^f[0-9]+$", names(new_dt), value = TRUE)
    stopifnot(length(new_feature_columns) == 2048L)
    X_new <- as.matrix(new_dt[, ..new_feature_columns])
    id_col <- if ("ID" %in% names(new_dt)) as.character(new_dt$ID) else seq_len(nrow(X_new))
    fn_col <- if ("filename" %in% names(new_dt)) as.character(new_dt$filename) else rep(NA_character_, nrow(X_new))
  } else if (!is.null(X_test)) {
    # no input file: predict the held-out test set created by the split
    X_new <- X_test
    id_col <- test_id_col
    fn_col <- test_fn_col
  } else {
    X_new <- X_train
    id_col <- as.character(feat_dt$ID)
    fn_col <- as.character(feat_dt$filename)
  }

  list(X_train = X_train, y_train = y_train,
       X_test = X_test, y_test = y_test,
       X_new = X_new,
       id_col = id_col, fn_col = fn_col,
       test_id_col = test_id_col, test_fn_col = test_fn_col,
       feature_columns = feature_columns)
}

# Command-line entry point.
run_classify <- function() {
  arguments <- commandArgs(trailingOnly = TRUE)
  options <- list(input = NULL, train_ratio = NULL, screening = "ttest",
                  transform = "none", seed = 2026L, labels = NULL)
  for (argument in arguments) {
    if (!grepl("^--[^=]+=", argument)) stop("Invalid option: ", argument)
    key <- sub("^--([^=]+)=.*$", "\\1", argument)
    value <- sub("^--[^=]+=", "", argument)
    if (identical(key, "input")) options$input <- value
    else if (identical(key, "labels")) options$labels <- value
    else if (identical(key, "train-ratio")) options$train_ratio <- as.numeric(value)
    else if (identical(key, "screening")) options$screening <- value
    else if (identical(key, "transform")) options$transform <- value
    else if (identical(key, "seed")) options$seed <- as.integer(value)
    else stop("Unknown option: ", key)
  }
  stopifnot(options$screening %in% c("ttest", "l1lr"),
            options$transform %in% c("none", "zscore", "rank_int"))

  if (!requireNamespace("data.table", quietly = TRUE)) stop("Install data.table first")
  if (!requireNamespace("quadprog", quietly = TRUE)) stop("Install quadprog first")
  if (!requireNamespace("TULIP", quietly = TRUE)) stop("Install TULIP first")

  data <- prepare_prop_data(input = options$input, train_ratio = options$train_ratio,
                            seed = options$seed, labels = options$labels)
  cat("Training PROP on", nrow(data$X_train), "samples (screening =", options$screening,
      ", transform =", options$transform, ")\n")
  if (!is.null(data$X_test)) {
    cat("Held-out test set:", nrow(data$X_test), "samples\n")
  }
  object <- prop_train(data$X_train, data$y_train, screening = options$screening,
                       seed = options$seed)
  cat("n_submodels =", object$n_submodels, "| n_screened =",
      length(object$p_sort_index), "\n")
  cat("Weights:", paste(sprintf("%.4f", object$weights), collapse = " "), "\n")

  transformed <- transform_train_test(data$X_train, data$X_new, options$transform)
  prediction <- prop_predict(object, transformed$test)
  out <- data.frame(ID = data$id_col, filename = data$fn_col, Class = prediction)
  cat("\nPredictions (class 1 = cataract only, class 2 = cataract + comorbidity):\n")
  print(out, row.names = FALSE)
  if (!is.null(data$y_test) && nrow(out) == length(data$y_test)) {
    cat("Test accuracy:", sprintf("%.4f", mean(prediction == data$y_test)), "\n")
  }
  cat("\nTip: source this script and call prepare_prop_data(input, train_ratio) to get X_train / y_train / X_test / X_new for your own split.\n")
}

# Run only when executed directly (not when sourced).
if (sys.nframe() == 0L) {
  run_classify()
}
