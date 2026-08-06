# PROP core implementation for the cataract_patient_dataset experiment.
#
# Algorithm: BH/t-test (or L1 logistic) screening -> three tiers of random
# subspaces -> DSDA base learners -> simplex-constrained QP weights -> weighted
# continuous DSDA scores.  Labels must be encoded as {1, 2}.

make_prop_data <- function(X, y) {
  stopifnot(is.matrix(X), all(y %in% c(1L, 2L)))
  list(
    n = length(y), p = ncol(X), data = X,
    data_1 = X[y == 1L, , drop = FALSE],
    data_2 = X[y == 2L, , drop = FALSE],
    labels = as.integer(y)
  )
}

transform_train_test <- function(X_train, X_test,
                                 method = c("none", "zscore", "rank_int")) {
  method <- match.arg(method)
  if (identical(method, "none")) {
    return(list(train = X_train, test = X_test))
  }
  if (identical(method, "zscore")) {
    mu <- colMeans(X_train)
    sigma <- apply(X_train, 2, sd)
    sigma[sigma < 1e-8] <- 1
    return(list(
      train = sweep(sweep(X_train, 2, mu, "-"), 2, sigma, "/"),
      test = sweep(sweep(X_test, 2, mu, "-"), 2, sigma, "/")
    ))
  }

  n_train <- nrow(X_train)
  eps <- 1 / (2 * n_train)
  Xtr <- matrix(0, nrow(X_train), ncol(X_train))
  Xte <- matrix(0, nrow(X_test), ncol(X_test))
  for (j in seq_len(ncol(X_train))) {
    Xtr[, j] <- qnorm((rank(X_train[, j], ties.method = "average") - 0.5) / n_train)
    u <- ecdf(X_train[, j])(X_test[, j])
    Xte[, j] <- qnorm(pmin(pmax(u, eps), 1 - eps))
  }
  list(train = Xtr, test = Xte)
}

screen_ttest <- function(NSC, data) {
  p_values <- vapply(seq_len(data$p), function(j) {
    tryCatch(t.test(data$data_2[, j], data$data_1[, j])$p.value,
             error = function(e) 1)
  }, numeric(1))
  significant <- which(p.adjust(p_values, method = "BH") < 0.05)
  selected <- if (length(significant) > NSC) {
    significant[order(p_values[significant])]
  } else {
    order(p_values)[seq_len(NSC)]
  }
  attr(selected, "n_significant") <- length(significant)
  selected
}

screen_l1lr <- function(NSC, data) {
  if (!requireNamespace("glmnet", quietly = TRUE)) {
    stop("screening='l1lr' requires the glmnet package")
  }
  fit <- glmnet::cv.glmnet(data$data, as.factor(data$labels), family = "binomial",
                            alpha = 1, nfolds = 5, type.measure = "class")
  beta <- abs(as.matrix(glmnet::coef.glmnet(fit, s = "lambda.min"))[-1, 1])
  nonzero <- which(beta > 0)
  selected <- if (length(nonzero) > NSC) {
    nonzero[order(beta[nonzero], decreasing = TRUE)]
  } else {
    order(beta, decreasing = TRUE)[seq_len(NSC)]
  }
  attr(selected, "n_significant") <- length(nonzero)
  selected
}

screen_variables <- function(NSC, data, method = c("ttest", "l1lr")) {
  method <- match.arg(method)
  if (identical(method, "ttest")) screen_ttest(NSC, data) else screen_l1lr(NSC, data)
}

sample_subspaces <- function(index_length, n_submodels) {
  sizes <- floor(c(0.2, 0.5, 0.8) * index_length)
  n_tier_1 <- floor(n_submodels / 3)
  n_tier_2 <- floor(2 * n_submodels / 3) - n_tier_1
  n_tier_3 <- n_submodels - n_tier_1 - n_tier_2
  # Preserve the original PROP ordering: all 20% submodels first, then 50%,
  # then 80%. The order matters because it determines the random-number stream.
  tier <- c(rep(1L, n_tier_1), rep(2L, n_tier_2), rep(3L, n_tier_3))
  lapply(tier, function(k) sample.int(index_length, size = sizes[k], replace = FALSE))
}

make_one_hot <- function(y) {
  Y <- matrix(0, nrow = 2 * length(y), ncol = 1)
  Y[seq(1, 2 * length(y) - 1, by = 2), 1] <- as.integer(y == 1L)
  Y[seq(2, 2 * length(y), by = 2), 1] <- as.integer(y == 2L)
  Y
}

make_prediction_matrix <- function(predictions) {
  n <- length(predictions[[1]])
  H <- matrix(0, nrow = 2 * n, ncol = length(predictions))
  for (m in seq_along(predictions)) {
    H[seq(1, 2 * n - 1, by = 2), m] <- as.integer(predictions[[m]] == 1L)
    H[seq(2, 2 * n, by = 2), m] <- as.integer(predictions[[m]] == 2L)
  }
  H
}

solve_prop_weights <- function(Y, H) {
  Dmat <- 2 * crossprod(H)
  M <- ncol(H)
  eigenvalues <- eigen(Dmat, only.values = TRUE)$values
  perturb <- max(max(eigenvalues) - M * min(eigenvalues), 0) / (M - 1)
  result <- quadprog::solve.QP(
    Dmat + diag(M) * perturb,
    2 * crossprod(H, Y),
    cbind(rep(1, M), diag(M)),
    c(1, rep(0, M)), meq = 1
  )
  as.numeric(result$solution)
}

fit_prop <- function(X, y, screening = c("ttest", "l1lr")) {
  screening <- match.arg(screening)
  if (!requireNamespace("TULIP", quietly = TRUE)) {
    stop("PROP requires the TULIP package (cv.dsda, dsda, predict.dsda)")
  }
  data <- make_prop_data(X, y)
  NSC <- floor(data$n / log(data$n))
  p_sort_index <- screen_variables(NSC, data, screening)
  X_screened <- X[, p_sort_index, drop = FALSE]
  n_submodels <- floor(3 * sqrt(data$n))
  subindex <- sample_subspaces(ncol(X_screened), n_submodels)
  X_submodels <- lapply(subindex, function(idx) X_screened[, idx, drop = FALSE])

  lambdas <- vapply(X_submodels, function(Xm) TULIP::cv.dsda(Xm, y = y)$lambda.min, numeric(1))
  models <- Map(function(Xm, lambda) TULIP::dsda(x = Xm, y = y, lambda = lambda),
                X_submodels, lambdas)
  train_predictions <- Map(function(model, Xm) TULIP::predict.dsda(model, Xm), models, X_submodels)
  H <- make_prediction_matrix(train_predictions)

  list(
    p_sort_index = p_sort_index,
    subindex = subindex,
    lambdas = lambdas,
    models = models,
    weights = solve_prop_weights(make_one_hot(y), H),
    n_significant = attr(p_sort_index, "n_significant"),
    n_submodels = n_submodels
  )
}

predict_prop <- function(object, X) {
  X_screened <- X[, object$p_sort_index, drop = FALSE]
  scores <- Map(function(model, idx) {
    as.numeric(cbind(1, X_screened[, idx, drop = FALSE]) %*% model$beta)
  }, object$models, object$subindex)
  weighted_score <- Reduce(`+`, Map(`*`, scores, object$weights))
  ifelse(weighted_score > 0, 2L, 1L)
}

# ---------------------------------------------------------------------------
# High-level convenience API (train / classify)
#
# prop_train()   fits PROP on a user-supplied training set and returns the
#                fitted object, including the learned ensemble `weights`.
# prop_predict() applies a fitted PROP object to new feature rows and returns
#                class labels 1 or 2.
#
# The packaged data uses labels 0/1; PROP itself uses 1/2, so convert with
# `y_train + 1L` (or `ifelse(y_train == 0, 1L, 2L)`) before calling prop_train().
# ---------------------------------------------------------------------------

prop_train <- function(X, y, screening = c("ttest", "l1lr"), seed = 2026L) {
  screening <- match.arg(screening)
  if (!is.null(seed)) set.seed(seed)
  fit_prop(X = X, y = y, screening = screening)
}

prop_predict <- function(object, X) {
  predict_prop(object, X)
}
