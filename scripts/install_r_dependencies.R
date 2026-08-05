#!/usr/bin/env Rscript

required <- c("data.table", "quadprog", "TULIP")
missing <- required[!vapply(required, requireNamespace, logical(1), quietly = TRUE)]
if (length(missing)) {
  install.packages(missing, repos = "https://cloud.r-project.org")
}
cat("Installed/available packages:", paste(required, collapse = ", "), "\n")
cat("Optional for --screening=l1lr: install.packages('glmnet')\n")
