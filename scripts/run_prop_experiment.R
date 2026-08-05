#!/usr/bin/env Rscript

# Reproduce the published PROP protocol from the packaged 212-sample cohort.

arguments <- commandArgs(trailingOnly = TRUE)
script_flag <- grep("^--file=", commandArgs(FALSE), value = TRUE)
script_path <- normalizePath(sub("^--file=", "", script_flag[[1]]), winslash = "/")
repo_root <- normalizePath(file.path(dirname(script_path), ".."), winslash = "/")
source(file.path(repo_root, "src", "PROP.R"))

options <- list(reps = 100L, seed_start = 714L, screening = "ttest", transform = "none")
for (argument in arguments) {
  if (!grepl("^--[^=]+=", argument)) stop("Invalid option: ", argument)
  key <- sub("^--([^=]+)=.*$", "\\1", argument)
  value <- sub("^--[^=]+=", "", argument)
  if (identical(key, "reps")) options$reps <- as.integer(value)
  else if (identical(key, "seed-start")) options$seed_start <- as.integer(value)
  else if (identical(key, "screening")) options$screening <- value
  else if (identical(key, "transform")) options$transform <- value
  else stop("Unknown option: ", key)
}
stopifnot(options$reps > 0, options$screening %in% c("ttest", "l1lr"),
          options$transform %in% c("none", "zscore", "rank_int"))

if (!requireNamespace("data.table", quietly = TRUE)) stop("Install data.table first")
if (!requireNamespace("quadprog", quietly = TRUE)) stop("Install quadprog first")
if (!requireNamespace("TULIP", quietly = TRUE)) stop("Install TULIP first")

features_csv <- file.path(repo_root, "data", "processed", "features_resnet50.csv")
labels_csv <- file.path(repo_root, "data", "processed", "cataract_labels.csv")
feat_dt <- data.table::fread(features_csv)
lab_dt <- data.table::fread(labels_csv)
stopifnot(identical(feat_dt$ID, lab_dt$ID), identical(feat_dt$filename, lab_dt$filename))

feature_columns <- grep("^f[0-9]+$", names(feat_dt), value = TRUE)
stopifnot(length(feature_columns) == 2048L, all(lab_dt$Y_cataract %in% c(0L, 1L)))
X_all <- as.matrix(feat_dt[, ..feature_columns])
y_all <- as.integer(lab_dt$Y_cataract) + 1L

split_stratified <- function(y, train_fraction = 2 / 3) {
  class_1 <- which(y == 1L)
  class_2 <- which(y == 2L)
  n_1 <- floor(length(class_1) * train_fraction)
  n_2 <- floor(length(class_2) * train_fraction)
  train <- c(sample(class_1, n_1), sample(class_2, n_2))
  list(train = train, test = setdiff(seq_along(y), train))
}

transform_tag <- switch(options$transform, none = "raw", zscore = "zscore", rank_int = "rankint")
setting <- sprintf("cataract_only_vs_plus_2_to_1_%s_%s", transform_tag, options$screening)
output_dir <- file.path(repo_root, "results", "reproduced")
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
output_file <- file.path(output_dir, sprintf("PROP_ResNet50_Setting_%s.csv", setting))

seeds <- seq.int(options$seed_start, length.out = options$reps)
summary <- data.frame(
  Replication = seq_len(options$reps), Seed = seeds,
  Time_Seconds = numeric(options$reps), Error_Rate = numeric(options$reps),
  N_Significant = integer(options$reps), Error_SD = rep(NA_real_, options$reps),
  stringsAsFactors = FALSE
)

for (i in seq_len(options$reps)) {
  set.seed(seeds[[i]])
  started <- Sys.time()
  split <- split_stratified(y_all)
  transformed <- transform_train_test(X_all[split$train, , drop = FALSE],
                                      X_all[split$test, , drop = FALSE], options$transform)
  object <- fit_prop(transformed$train, y_all[split$train], options$screening)
  prediction <- predict_prop(object, transformed$test)
  summary$Error_Rate[[i]] <- mean(prediction != y_all[split$test])
  summary$Time_Seconds[[i]] <- as.numeric(difftime(Sys.time(), started, units = "secs"))
  summary$N_Significant[[i]] <- object$n_significant
  cat(sprintf("[rep %3d/%3d] seed=%d err=%.4f n_sig=%d\n", i, options$reps,
              seeds[[i]], summary$Error_Rate[[i]], summary$N_Significant[[i]]))
}

overall <- data.frame(
  Replication = NA_integer_, Seed = NA_integer_,
  Time_Seconds = sum(summary$Time_Seconds),
  Error_Rate = mean(summary$Error_Rate),
  N_Significant = NA_integer_, Error_SD = sd(summary$Error_Rate)
)
result <- rbind(summary, overall)
data.table::fwrite(result, output_file)
cat(sprintf("\nSaved %s\nMean error: %.4f | SD: %.4f\n", output_file,
            mean(summary$Error_Rate), sd(summary$Error_Rate)))
